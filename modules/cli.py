# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import logging
import yaml
import json
import time

import multistorageclient as msc
from omegaconf import OmegaConf

from captioning.factory import create_captioner
from data_processing import run_alignment
from generation.factory import create_generator
from generation.utils import generate_with_retries
from verification.llm_question_generator import LLMQuestionGenerator
from verification.vlm_verifier import VLMVerifier
from verification.core import AttributeVerifier
from verification.hallucination_checker import HallucinationChecker
from aug_utils.common import (
    validate_config_structure,
    validate_sample_data_availability,
)
from aug_utils.nvcf import load_secrets, NVCFProgressTracker


def validate_environment(logger: logging.Logger):
    """Log which optional env vars are set."""
    optional_env_vars = [
        "LOG_LEVEL",
        "VLM_API_KEY",
        "LLM_API_KEY",
        "VLM_ENDPOINT_URL",
        "VLM_ENDPOINT_MODEL",
        "LLM_ENDPOINT_URL",
        "LLM_ENDPOINT_MODEL",
        "COSMOS_ENDPOINT_URL",
        "COSMOS_ENDPOINT_MODEL",
        "COSMOS_PREDICT_ENDPOINT_URL",
        "IMAGE_EDIT_ENDPOINT_URL",
        "IMAGE_EDIT_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "AWS_ENDPOINT_URL",
        "AWS_S3_BUCKET",
        "AWS_S3_ADDRESSING_STYLE",
    ]
    for var in optional_env_vars:
        if not os.getenv(var):
            logger.debug(f"Environment variable {var} is not set")
    logger.info(
        "Environment: keys from terminal, endpoint/model from config when unset."
    )


# ---------------------------------------------------------------------------
# Evaluator initialisation helpers
# ---------------------------------------------------------------------------


def _init_evaluators(config, logger):
    """Initialise evaluators from the ``evaluators`` list in the config.

    Returns (hallucination_checker, attribute_verifier).
    """
    hallucination_checker = None
    llm_question_generator = None
    vlm_verifier = None
    av_config_obj = None  # keep reference for retry/extra_questions later

    evaluators = config.get("evaluators") or []
    if hasattr(evaluators, "__iter__") and not isinstance(evaluators, (str, dict)):
        pass  # already a list
    else:
        evaluators = []

    endpoints = config.get("endpoints") or {}
    # Resolve endpoint dicts from Pydantic models if needed
    if hasattr(endpoints, "model_dump"):
        endpoints = endpoints.model_dump()

    for entry in evaluators:
        # Support both Pydantic model and raw dict
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump(exclude_none=False)

        if entry.get("hallucination_check") is not None:
            hc = entry["hallucination_check"]
            if hc.get("enabled", True):
                logger.info("Initializing hallucination checker...")
                hallucination_checker = HallucinationChecker(
                    params=hc.get("params", {}),
                    logger=logger,
                )
                logger.info("Hallucination checker initialized successfully")

        elif entry.get("attribute_verification") is not None:
            av = entry["attribute_verification"]
            av_config_obj = av
            if av.get("enabled", True):
                logger.info(
                    "Initializing attribute verification (question generator)..."
                )
                qg = av.get("question_generation", {})
                llm_ep = endpoints.get("llm") or {}
                llm_question_generator = LLMQuestionGenerator.from_config(
                    config_params=qg,
                    system_prompt=qg.get("system_prompt", ""),
                    endpoint=(os.getenv("LLM_ENDPOINT_URL") or llm_ep.get("url", "")),
                    model=(os.getenv("LLM_ENDPOINT_MODEL") or llm_ep.get("model", "")),
                    logger=logger,
                )
                # Initialize VLM verifier from inline vlm_verification config
                # if present, otherwise fall back to endpoints.vlm with defaults.
                if vlm_verifier is None:
                    inline_vlm = av.get("vlm_verification")
                    vlm_ep = endpoints.get("vlm") or {}
                    vlm_endpoint = os.getenv("VLM_ENDPOINT_URL") or vlm_ep.get(
                        "url", ""
                    )
                    vlm_model = os.getenv("VLM_ENDPOINT_MODEL") or vlm_ep.get(
                        "model", ""
                    )
                    if inline_vlm is not None and vlm_endpoint:
                        logger.info(
                            "Initializing VLM verifier from attribute_verification.vlm_verification..."
                        )
                        vlm_verifier = VLMVerifier.from_config(
                            config_params=inline_vlm,
                            endpoint=vlm_endpoint,
                            model=vlm_model,
                            logger=logger,
                        )
                    elif vlm_endpoint:
                        logger.info(
                            "Initializing VLM verifier from endpoints.vlm with defaults..."
                        )
                        vlm_verifier = VLMVerifier.from_config(
                            config_params={"system_prompt": "", "parameters": {}},
                            endpoint=vlm_endpoint,
                            model=vlm_model,
                            logger=logger,
                        )

        elif entry.get("vlm_verification") is not None:
            vv = entry["vlm_verification"]
            logger.info("Initializing VLM verifier...")
            vlm_ep = endpoints.get("vlm") or {}
            vlm_verifier = VLMVerifier.from_config(
                config_params=vv,
                endpoint=(os.getenv("VLM_ENDPOINT_URL") or vlm_ep.get("url", "")),
                model=(os.getenv("VLM_ENDPOINT_MODEL") or vlm_ep.get("model", "")),
                logger=logger,
            )

    attribute_verifier = None
    if llm_question_generator is not None and vlm_verifier is not None:
        attribute_verifier = AttributeVerifier(
            question_generator=llm_question_generator,
            vlm_verifier=vlm_verifier,
            logger=logger,
        )
        logger.info("Attribute verification initialized successfully")

    return hallucination_checker, attribute_verifier, av_config_obj


def _write_prompt_file(prompt_path: str, prompt: str, logger: logging.Logger) -> bool:
    """Persist the current prompt so retries keep metadata in sync."""
    try:
        with msc.open(prompt_path, "w") as f:
            f.write(prompt)
        logger.info(f"Prompt written to {prompt_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write prompt file {prompt_path}: {e}")
        return False


def _generate_prompt(
    sample: dict,
    captioner,
    logger: logging.Logger,
    rgb_path: str | None = None,
    write_prompt: bool = True,
):
    """Generate a prompt for a sample and optionally persist it."""
    if captioner is None:
        logger.error("No captioner configured to generate a prompt")
        return None

    logger.info("Running captioning...")
    try:
        prompt = captioner.get_caption(rgb_path)
        logger.debug(f"Caption: {prompt}")
        logger.info("Captioning completed successfully")
    except Exception as e:
        logger.error(f"Failed to generate caption for {rgb_path}: {e}")
        return None

    if prompt is None:
        logger.error("No prompt was produced from captioning")
        return None

    if write_prompt and not _write_prompt_file(
        sample["output"]["caption"], prompt, logger
    ):
        return None

    return prompt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown_args = parser.parse_known_args()

    # Initialize logger
    logger = logging.getLogger(__name__)
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print()
    logger.info("=" * 80)
    logger.info("PAIDF Augmentation")
    logger.info(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    load_secrets(logger)

    # ---- Load & validate config ----
    try:
        with msc.open(args.config, "r") as f:
            base_config = yaml.safe_load(f)
            conf = OmegaConf.create(base_config)
            OmegaConf.set_struct(conf, False)
            if unknown_args:
                logger.info(f"Applying CLI overrides: {unknown_args}")
                conf.merge_with_dotlist(unknown_args)
            config_dict = OmegaConf.to_container(conf, resolve=True)
    except Exception as e:
        logger.error(
            f"Failed to read config file {args.config} or apply overrides: {e}"
        )
        return

    validate_environment(logger)

    pipeline_config = validate_config_structure(config_dict, logger)
    if pipeline_config is None:
        logger.error(
            "Configuration validation failed. Please fix the configuration file."
        )
        return

    # For convenience keep a plain dict view for sections that still need it
    config = config_dict
    config_dir = os.path.dirname(args.config)

    # ---- Initialize captioner ----
    captioner = None
    if config.get("captioning"):
        logger.info("Initializing captioner...")
        try:
            captioner = create_captioner(config, logger, config_dir=config_dir)
            logger.info("Captioner initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize captioner: {e}")
            return

    # ---- Initialize evaluators ----
    hallucination_checker, attribute_verifier, av_config_obj = _init_evaluators(
        config, logger
    )

    # ---- Initialize generator ----
    generator = None
    if config.get("augmentation"):
        model_name = config["augmentation"]["model"]["name"]
        executor_type = config["augmentation"]["model"].get("executor_type", "local")
        try:
            generator = create_generator(config, logger)
        except ModuleNotFoundError:
            if "cosmos" in model_name and executor_type == "local":
                logger.error(
                    "COSMOS local execution is not supported in this environment. "
                    "Please run with cosmos local dependencies or use a different executor."
                )
                logger.debug("Import error while creating generator", exc_info=True)
                return
            raise
        logger.info("Generator initialized successfully")

    overwrite_caption = os.getenv("OVERWRITE_CAPTION", "true")

    # ---- NVCF progress tracker ----
    tracker = NVCFProgressTracker(logger=logger)
    data_samples = config["data"]
    total_samples = len(data_samples)
    samples_processed = 0
    models_fetched = False

    # ---- Process samples ----
    for sample in data_samples:
        logger.info("=" * 80)
        sample_inputs = sample.get("inputs") or {}
        rgb_path = sample_inputs.get("rgb") if isinstance(sample_inputs, dict) else None
        logger.info(f"Processing sample: {rgb_path or 'N/A (text2world)'}")
        try:
            logger.debug(f"Processing sample: {sample}")

            if not validate_sample_data_availability(sample, config, logger):
                logger.error("Sample validation failed, skipping sample")
                continue

            output_media_path = sample["output"]["video"]

            prompt = None
            control_inputs = {}
            natural_caption_from_vlm = None

            # ---- Captioning ----
            caption_exists = msc.is_file(sample["output"]["caption"])
            caption_content = None
            if caption_exists:
                try:
                    with msc.open(sample["output"]["caption"], "r") as f:
                        caption_content = f.read()
                    logger.debug(f"Caption file exists: {sample['output']['caption']}")
                except Exception as e:
                    logger.error(
                        f"Failed to read caption file {sample['output']['caption']}: {e}"
                    )
                    caption_exists = False

            if not caption_exists or overwrite_caption == "true":
                prompt = _generate_prompt(sample, captioner, logger, rgb_path=rgb_path)
                if prompt is None:
                    continue
            else:
                prompt = caption_content
                logger.debug("Using existing prompt from file")

            # ---- Generation + Verification retry loop ----
            if generator is not None:
                hallucination_enabled = hallucination_checker is not None
                verification_enabled = attribute_verifier is not None

                max_retries = pipeline_config.pipeline.retry
                request_timeout = pipeline_config.pipeline.request_timeout
                if max_retries > 0:
                    logger.info(
                        f"Retry enabled (pipeline.retry): up to {max_retries} "
                        f"retries on evaluator failure (seed re-roll) or "
                        f"generation/endpoint failure (exponential backoff from "
                        f"{request_timeout:.0f}s)."
                    )

                # Seed handling
                aug_params = config.get("augmentation", {}).get("parameters", {})
                regenerate_caption_on_retry = (
                    pipeline_config.pipeline.regenerate_caption_on_retry
                )
                original_seed = aug_params.get("seed")
                if original_seed in (None, "None", "none"):
                    if hasattr(generator, "seed") and generator.seed is not None:
                        original_seed = generator.seed
                    elif (
                        hasattr(generator, "executor")
                        and hasattr(generator.executor, "seed")
                        and generator.executor.seed is not None
                    ):
                        original_seed = generator.executor.seed
                    else:
                        original_seed = int(time.time())

                current_seed = original_seed
                hallucination_results = None
                verification_results = None
                passed_hallucination_check = True
                generation_succeeded = False
                max_attempts = max_retries + 1

                for attempt in range(max_attempts):
                    if attempt > 0:
                        current_seed = original_seed + attempt
                        logger.info("=" * 80)
                        logger.info(
                            f"RETRY ATTEMPT {attempt}/{max_retries}: seed {current_seed}"
                        )
                        logger.info("=" * 80)
                        if models_fetched and not os.environ.get("HF_HUB_OFFLINE"):
                            os.environ["HF_HUB_OFFLINE"] = "1"
                            logger.info("Set HF_HUB_OFFLINE=1 for retry (cache warm)")
                        if hasattr(generator, "seed"):
                            generator.seed = current_seed
                        elif hasattr(generator, "executor") and hasattr(
                            generator.executor, "seed"
                        ):
                            generator.executor.seed = current_seed

                        if regenerate_caption_on_retry and captioner is not None:
                            logger.info("Regenerating prompt for retry...")
                            prompt = _generate_prompt(
                                sample, captioner, logger, rgb_path=rgb_path
                            )
                            if prompt is None:
                                logger.error("Retry captioning failed; aborting sample")
                                break

                    print()
                    logger.info(f"Running generation (seed: {current_seed})...")

                    if prompt is None:
                        logger.error("Prompt is required for generation")
                        break

                    # Load control inputs from augmentation.modalities
                    try:
                        control_inputs = {}
                        modalities_cfg = (
                            config.get("augmentation", {}).get("modalities") or {}
                        )
                        if (
                            sample_inputs
                            and "controls" in sample_inputs
                            and sample_inputs["controls"]
                        ):
                            for mod_key in ("edge", "depth", "seg", "vis"):
                                weight = modalities_cfg.get(mod_key)
                                if (
                                    weight is not None
                                    and mod_key in sample_inputs["controls"]
                                ):
                                    control_inputs[mod_key] = sample_inputs["controls"][
                                        mod_key
                                    ]
                            if not control_inputs:
                                control_inputs = {
                                    k: v
                                    for k, v in sample_inputs["controls"].items()
                                    if v is not None
                                    and str(v) not in ("None", "none", "null")
                                }
                            logger.debug(
                                f"Control inputs: {list(control_inputs.keys())}"
                            )
                    except (TypeError, KeyError, ValueError, AttributeError) as e:
                        logger.error(
                            f"Failed to prepare control inputs -> {output_media_path}: {e}"
                        )
                        break

                    # Generate, re-calling the endpoint on failure up to the
                    # pipeline.retry budget. Each call is bounded by request_timeout,
                    # which is also the exponential-backoff base between retries.
                    success, output_path, gen_elapsed = generate_with_retries(
                        generator,
                        prompt,
                        rgb_path,
                        control_inputs,
                        output_media_path,
                        max_retries,
                        request_timeout,
                        logger,
                    )
                    if not success:
                        logger.error(
                            f"Generation failed after {max_retries + 1} attempt(s) "
                            f"-> {output_media_path}"
                        )
                        break
                    models_fetched = True
                    generation_succeeded = True
                    sample["output"]["video"] = output_path
                    logger.info(
                        f"Generation done in {gen_elapsed / 60:.2f} min, "
                        f"saved to: {output_path}"
                    )

                    # ---- Hallucination check ----
                    passed_hallucination_check = True
                    if hallucination_enabled and rgb_path is not None:
                        print()
                        logger.info("Running hallucination check...")
                        try:
                            # Get threshold from evaluator config
                            hc_threshold = 0.682
                            for entry in config.get("evaluators") or []:
                                if hasattr(entry, "model_dump"):
                                    entry = entry.model_dump(exclude_none=False)
                                hc = entry.get("hallucination_check")
                                if hc:
                                    hc_threshold = hc.get("threshold", 0.682)

                            check_passed, details = (
                                hallucination_checker.check_hallucination(
                                    original_video_path=rgb_path,
                                    augmented_video_path=sample["output"]["video"],
                                )
                            )
                            score = details.get("score", 0.0)
                            passed_hallucination_check = (
                                check_passed and score >= hc_threshold
                            )
                            hallucination_results = {
                                "passed": passed_hallucination_check,
                                "score": score,
                                "threshold": hc_threshold,
                                "details": details,
                                "attempt": attempt + 1,
                                "seed_used": current_seed,
                            }
                            if not passed_hallucination_check:
                                logger.warning(
                                    f"Hallucination check failed (score={score:.4f})"
                                )
                                if attempt < max_retries:
                                    continue
                                else:
                                    logger.warning(
                                        "Exhausted retries for hallucination check"
                                    )
                            else:
                                logger.info(
                                    f"Hallucination check passed (score={score:.4f})"
                                )
                        except Exception as e:
                            logger.error(f"Hallucination check error: {e}")
                            hallucination_results = {
                                "passed": False,
                                "error": str(e),
                                "attempt": attempt + 1,
                                "seed_used": current_seed,
                            }
                            passed_hallucination_check = False

                    # ---- Attribute verification ----
                    if passed_hallucination_check and verification_enabled:
                        print()
                        logger.info("Running attribute verification...")
                        try:
                            llm_cfg = (config.get("captioning") or {}).get("llm") or {}
                            variables = llm_cfg.get("variables") or {}
                            verification_values = llm_cfg.get("verification_values")
                            verification_options = llm_cfg.get("verification_options")

                            if verification_values is not None:
                                selected_variables = {
                                    k: (v[0] if isinstance(v, list) and v else v)
                                    for k, v in verification_values.items()
                                }
                            else:
                                selected_variables = {
                                    k: (v[0] if isinstance(v, list) and v else v)
                                    for k, v in variables.items()
                                }

                            variable_options = (
                                verification_options
                                if verification_options is not None
                                else variables
                            )
                            exclude_variables = set(
                                (av_config_obj or {}).get("exclude_variables") or []
                            )
                            if exclude_variables:
                                selected_variables = {
                                    k: v
                                    for k, v in selected_variables.items()
                                    if k not in exclude_variables
                                }
                                variable_options = {
                                    k: v
                                    for k, v in variable_options.items()
                                    if k not in exclude_variables
                                }
                            extra_questions = (av_config_obj or {}).get(
                                "extra_questions"
                            ) or []

                            passed_attr, ver_details = (
                                attribute_verifier.verify_video_attributes(
                                    video_path=sample["output"]["video"],
                                    selected_variables=selected_variables,
                                    variable_options=variable_options,
                                    extra_questions=extra_questions,
                                )
                            )
                            verification_results = {
                                "passed": passed_attr,
                                "details": ver_details,
                                "attempt": attempt + 1,
                                "seed_used": current_seed,
                            }

                            if passed_attr:
                                # Natural caption generation on pass
                                gen_cap = (av_config_obj or {}).get(
                                    "generate_natural_caption_on_pass", False
                                )
                                cap_attrs = selected_variables
                                if gen_cap and attribute_verifier and cap_attrs:
                                    attrs_text = "; ".join(
                                        f"{k.replace('_', ' ')}: {v}"
                                        for k, v in cap_attrs.items()
                                    )
                                    nat_cfg = (av_config_obj or {}).get(
                                        "natural_caption"
                                    ) or {}
                                    user_tpl = nat_cfg.get("user_prompt_template") or (
                                        "This multi-view image shows a person with the following clothing: {attributes_text}. "
                                        "Write one natural sentence describing the person's appearance."
                                    )
                                    user_prompt = user_tpl.format(
                                        attributes_text=attrs_text
                                    )
                                    natural_caption_from_vlm = attribute_verifier.vlm_verifier.generate_description(
                                        sample["output"]["video"],
                                        user_prompt,
                                        system_prompt_override=nat_cfg.get(
                                            "system_prompt"
                                        ),
                                    )
                                    if natural_caption_from_vlm:
                                        logger.info(
                                            "Generated natural caption from VLM."
                                        )
                                logger.info("All attribute checks passed.")
                                break
                            else:
                                logger.warning("Some attribute checks failed.")
                                if attempt < max_retries:
                                    logger.info(
                                        f"Retrying (attempt {attempt + 2}/{max_attempts})"
                                    )
                                else:
                                    logger.warning(
                                        "Exhausted retries for attribute verification"
                                    )
                        except Exception as e:
                            logger.error(f"Attribute verification error: {e}")
                            verification_results = {
                                "passed": False,
                                "error": str(e),
                                "attempt": attempt + 1,
                                "seed_used": current_seed,
                            }
                            break
                    else:
                        if not passed_hallucination_check and attempt < max_retries:
                            continue
                        else:
                            break

                # Restore seed
                if original_seed is not None:
                    if hasattr(generator, "seed"):
                        generator.seed = original_seed
                    elif hasattr(generator, "executor") and hasattr(
                        generator.executor, "seed"
                    ):
                        generator.executor.seed = original_seed
            else:
                logger.warning("No generator configured, skipping generation")
                hallucination_results = None
                verification_results = None
                generation_succeeded = False

            # A generator ran but produced no output after exhausting retries —
            # log it and skip output/metadata so it isn't counted as completed.
            if generator is not None and not generation_succeeded:
                logger.error(f"Sample failed: {rgb_path or 'unknown'}")
                continue

            # ---- Data processing (alignment, ...) ----
            alignment_params = None
            dp_model = pipeline_config.data_processing
            align_model = dp_model.alignment if dp_model else None
            if align_model is not None:
                if not generation_succeeded:
                    logger.warning(
                        "Skipping alignment: generation did not succeed this run"
                    )
                elif not rgb_path:
                    logger.warning("Skipping alignment: no reference (rgb) input")
                else:
                    logger.info("Running data processing: alignment...")
                    alignment_params = run_alignment(
                        ref_path=rgb_path,
                        align_path=sample["output"]["video"],
                        dest_path=sample["output"]["video"],
                        config=align_model.model_dump(),
                        logger=logger,
                    )

            # ---- Write metadata ----
            try:
                llm_cfg = (config.get("captioning") or {}).get("llm") or {}
                metadata_selections = (
                    llm_cfg.get("verification_values") or llm_cfg.get("variables") or {}
                )
                metadata_selections = {
                    k: (v[0] if isinstance(v, list) and v else v)
                    for k, v in metadata_selections.items()
                }

                metadata = {
                    "prompt": prompt,
                    "selections": metadata_selections,
                    "caption_path": sample["output"]["caption"],
                    "output_media_path": sample["output"]["video"],
                    "input_media_path": rgb_path,
                    "control_media": control_inputs,
                }
                if hallucination_results is not None:
                    metadata["hallucination_check"] = hallucination_results
                if verification_results is not None:
                    metadata["attribute_verification"] = verification_results
                if natural_caption_from_vlm:
                    metadata["natural_caption"] = natural_caption_from_vlm
                if alignment_params:
                    metadata["alignment"] = alignment_params

                with msc.open(sample["output"]["metadata"], "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Metadata written to {sample['output']['metadata']}")

                # Write evaluation file if path specified
                eval_path = sample["output"].get("evaluation")
                if eval_path and (
                    hallucination_results is not None
                    or verification_results is not None
                ):
                    try:
                        eval_output = {}
                        if hallucination_results is not None:
                            eval_output["hallucination_check"] = hallucination_results
                        if verification_results is not None:
                            eval_output["attribute_verification"] = verification_results
                        with msc.open(eval_path, "w") as f:
                            json.dump(eval_output, f, indent=2)
                        logger.info(f"Evaluation results written to {eval_path}")
                    except Exception as e:
                        logger.error(f"Failed to write evaluation results: {e}")

            except Exception as e:
                logger.error(f"Failed to write metadata: {e}")
                continue

            logger.info("=" * 80)
            samples_processed += 1
            tracker.update(
                percent=(samples_processed / total_samples) * 100,
                metadata={
                    "lastProcessedSample": (sample.get("inputs") or {}).get(
                        "rgb", "unknown"
                    )
                },
            )

        except Exception as e:
            logger.error(f"Unexpected error processing sample: {e}")
            continue

    # samples_processed counts only fully successful samples, so any shortfall
    # means a sample failed (e.g. generation exhausted its retries); exit
    # non-zero without marking the task complete.
    if samples_processed < total_samples:
        logger.error(
            f"{total_samples - samples_processed}/{total_samples} sample(s) "
            f"failed; pipeline aborted without completion."
        )
        sys.exit(1)

    tracker.complete(metadata={"totalSamplesProcessed": samples_processed})


if __name__ == "__main__":
    main()
