# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import logging

from aug_utils.schema.augmentation import ModelNameEnum
from .base import BaseGenerator
from .cosmos_generator import CosmosGenerator
from .cosmos_predict_generator import CosmosPredictGenerator
from .image_edit_generator import ImageEditGenerator


def create_generator(config, logger: logging.Logger) -> BaseGenerator:
    """Create appropriate generator based on the ``augmentation`` configuration.

    Dispatches on ``augmentation.model.name``:
    - ``cosmos-transfer2.5`` → :class:`CosmosGenerator`
    - ``cosmos-predict``     → :class:`CosmosPredictGenerator` (stub)
    - ``image-edit``         → :class:`ImageEditGenerator`

    Args:
        config: Full pipeline config (PipelineConfig model or raw dict).
        logger: Logger instance.

    Returns:
        BaseGenerator: Configured generator instance.

    Raises:
        ValueError: If augmentation config is missing or model name is unknown.
    """
    # Support both Pydantic model and raw dict
    if hasattr(config, "augmentation"):
        aug = config.augmentation
        endpoints = config.endpoints
        if aug is None:
            raise ValueError("'augmentation' section is required to create a generator")
        model_name = (
            aug.model.name.value
            if hasattr(aug.model.name, "value")
            else str(aug.model.name)
        )
        executor_type = (
            aug.model.executor_type.value
            if hasattr(aug.model.executor_type, "value")
            else str(aug.model.executor_type)
        )
        params = (
            aug.parameters.model_dump()
            if hasattr(aug.parameters, "model_dump")
            else aug.parameters
        )
        modalities_cfg = (
            aug.modalities.model_dump()
            if aug.modalities and hasattr(aug.modalities, "model_dump")
            else (aug.modalities or {})
        )
        local_params = (
            aug.local_parameters.model_dump()
            if aug.local_parameters and hasattr(aug.local_parameters, "model_dump")
            else {}
        )
        model_version = aug.model.version
        endpoints_dict = (
            endpoints.model_dump() if hasattr(endpoints, "model_dump") else endpoints
        )
        request_timeout = config.pipeline.request_timeout
    else:
        aug_dict = config.get("augmentation")
        if not aug_dict:
            raise ValueError("'augmentation' section is required to create a generator")
        model_name = aug_dict["model"]["name"]
        executor_type = aug_dict["model"].get("executor_type", "local")
        params = aug_dict.get("parameters", {})
        modalities_cfg = aug_dict.get("modalities") or {}
        local_params = aug_dict.get("local_parameters") or {}
        model_version = aug_dict["model"].get("version")
        endpoints_dict = config.get("endpoints", {})
        request_timeout = (config.get("pipeline") or {}).get("request_timeout")

    if model_name == ModelNameEnum.COSMOS_TRANSFER.value:
        return _create_cosmos_transfer(
            params,
            modalities_cfg,
            local_params,
            endpoints_dict,
            executor_type,
            model_version,
            logger,
        )
    elif model_name == ModelNameEnum.COSMOS_PREDICT.value:
        return _create_cosmos_predict(
            params,
            local_params,
            endpoints_dict,
            executor_type,
            model_version,
            logger,
        )
    elif model_name == ModelNameEnum.IMAGE_EDIT.value:
        return _create_image_edit(
            params,
            endpoints_dict,
            request_timeout,
            logger,
        )
    else:
        valid_models = ", ".join(m.value for m in ModelNameEnum)
        raise ValueError(
            f"Unknown augmentation model: '{model_name}'. Valid models: {valid_models}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_seed(seed_value, logger: logging.Logger) -> int:
    """Resolve a seed value, generating one from time if None."""
    if seed_value is None or seed_value == "None" or seed_value == "none":
        seed = int(time.time())
        logger.debug(f"Generated random seed: {seed}")
        return seed
    return int(seed_value)


def _create_cosmos_transfer(
    params,
    modalities_cfg,
    local_params,
    endpoints_dict,
    executor_type,
    model_version,
    logger,
) -> CosmosGenerator:
    cosmos_ep = endpoints_dict.get("cosmos_transfer") or {}
    endpoint = (
        os.getenv("COSMOS_ENDPOINT_URL")
        or (
            cosmos_ep.get("url")
            if isinstance(cosmos_ep, dict)
            else getattr(cosmos_ep, "url", None)
        )
        or "http://localhost:30002/"
    )
    seed = _resolve_seed(params.get("seed"), logger)

    # Extract modality weights (numeric) vs prompt fields (string)
    modality_weights = {}
    for key in ("edge", "depth", "seg", "vis"):
        val = modalities_cfg.get(key)
        if val is not None:
            modality_weights[key] = float(val)

    modality_names = list(modality_weights.keys())

    # Gather any extra executor-specific params
    executor_specific = dict(local_params)
    seg_control_prompt = modalities_cfg.get("seg_control_prompt")
    if seg_control_prompt:
        executor_specific["seg_control_prompt"] = seg_control_prompt
    return CosmosGenerator(
        executor_type=executor_type,
        endpoint=endpoint,
        sigma=params.get("sigma", 90),
        seed=seed,
        guidance=params.get("guidance", 3),
        num_steps=params.get("num_steps", 35),
        modalities=modality_names,
        weights=modality_weights,
        positive_prompt=modalities_cfg.get("positive_prompt", ""),
        negative_prompt=modalities_cfg.get("negative_prompt", ""),
        inference_name=params.get("inference_name", "cosmos_transfer_inference"),
        logger=logger,
        model_version=model_version,
        **executor_specific,
    )


_PREDICT_MODEL_MAP = {
    "2B": "2B/post-trained",
    "2B-distilled": "2B/distilled",
    "14B": "14B/post-trained",
}


def _create_cosmos_predict(
    params, local_params, endpoints_dict, executor_type, model_version, logger
) -> BaseGenerator:
    if executor_type != "local":
        raise ValueError(
            f"Cosmos Predict currently only supports executor_type='local', "
            f"got '{executor_type}'"
        )
    predict_ep = endpoints_dict.get("cosmos_predict") or {}
    endpoint = (
        os.getenv("COSMOS_PREDICT_ENDPOINT_URL")
        or (
            predict_ep.get("url")
            if isinstance(predict_ep, dict)
            else getattr(predict_ep, "url", None)
        )
        or ""
    )
    seed = _resolve_seed(params.get("seed"), logger)

    # Resolve model name for --model CLI arg
    predict_model = None
    if model_version:
        predict_model = _PREDICT_MODEL_MAP.get(model_version, model_version)

    inference_type_raw = params.get("inference_type", "video2world")
    inference_type = (
        inference_type_raw.value
        if hasattr(inference_type_raw, "value")
        else str(inference_type_raw)
    )

    return CosmosPredictGenerator(
        endpoint=endpoint,
        inference_type=inference_type,
        num_output_frames=params.get("num_output_frames"),
        num_steps=params.get("num_steps", 35),
        enable_autoregressive=params.get("enable_autoregressive"),
        chunk_size=params.get("chunk_size"),
        chunk_overlap=params.get("chunk_overlap"),
        seed=seed,
        guidance=params.get("guidance", 3.0),
        inference_name=params.get("inference_name", "cosmos_predict_inference"),
        resolution=params.get("resolution", "none"),
        model=predict_model,
        logger=logger,
        offload_tokenizer=params.get("offload_tokenizer"),
        offload_text_encoder=params.get("offload_text_encoder"),
        **dict(local_params),
    )


def _create_image_edit(
    params, endpoints_dict, request_timeout, logger
) -> ImageEditGenerator:
    image_edit_ep = endpoints_dict.get("image_edit") or {}
    endpoint = (
        os.getenv("IMAGE_EDIT_ENDPOINT_URL")
        or (
            image_edit_ep.get("url")
            if isinstance(image_edit_ep, dict)
            else getattr(image_edit_ep, "url", None)
        )
        or ""
    )
    if not endpoint:
        raise ValueError("Image edit endpoint not configured")

    seed = _resolve_seed(params.get("seed"), logger)
    api_key = os.getenv("IMAGE_EDIT_API_KEY")
    model = (
        image_edit_ep.get("model")
        if isinstance(image_edit_ep, dict)
        else getattr(image_edit_ep, "model", None)
    ) or ""

    # Honor explicit `negative_prompt: null` in YAML (disables CFG) vs absent
    # key (use the canonical " " default that enables CFG with no real
    # negative). The schema default is " ", so unset YAMLs get CFG.
    negative_prompt = params["negative_prompt"] if "negative_prompt" in params else " "

    return ImageEditGenerator(
        endpoint=endpoint,
        num_inference_steps=params.get("num_inference_steps", 50),
        guidance_scale=params.get("guidance_scale", 1.0),
        seed=seed,
        model=model or None,
        api_key=api_key,
        negative_prompt=negative_prompt,
        timeout=request_timeout,
        logger=logger,
    )
