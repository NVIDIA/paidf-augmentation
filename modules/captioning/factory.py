# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import logging

from aug_utils.prompt_loader import load_system_prompt_from_config
from captioning.base import BaseCaptioner
from captioning.vlm_captioner import VLMCaptioner
from captioning.llm_captioner import LLMCaptioner
from captioning.vlm_llm_captioner import VLMLLMCaptioner
from captioning.text_captioner import TextCaptioner
from captioning.file_captioner import FileCaptioner


def create_captioner(
    config, logger: logging.Logger, config_dir: str = None
) -> BaseCaptioner:
    """Create appropriate captioner based on configuration.

    Captioning type is inferred from which sub-sections are present under
    ``captioning``:

    - ``captioning.llm.text`` is set    → TextCaptioner
    - ``captioning.llm.file_path`` is set → FileCaptioner
    - Both ``vlm`` and ``llm``          → VLMLLMCaptioner
    - Only ``vlm``                      → VLMCaptioner
    - Only ``llm``                      → LLMCaptioner

    Args:
        config: Full pipeline config (PipelineConfig model or raw dict).
        logger: Logger instance.

    Returns:
        BaseCaptioner: Configured captioner instance.

    Raises:
        ValueError: If configuration is invalid or incomplete.
    """
    # Support both Pydantic model and raw dict
    if hasattr(config, "captioning"):
        captioning_cfg = config.captioning
        endpoints_cfg = config.endpoints
        # Convert to dicts for downstream constructors that still expect dicts
        captioning_dict = captioning_cfg.model_dump() if captioning_cfg else {}
        endpoints_dict = endpoints_cfg.model_dump() if endpoints_cfg else {}
    else:
        captioning_dict = config.get("captioning", {})
        endpoints_dict = config.get("endpoints", {})

    vlm_config = captioning_dict.get("vlm")
    llm_config = captioning_dict.get("llm")
    has_vlm = vlm_config is not None
    has_llm = llm_config is not None

    # Infer captioner type from which fields are set
    llm_text = (llm_config or {}).get("text")
    llm_file_path = (llm_config or {}).get("file_path")

    # text and file_path are mutually exclusive
    if llm_text and llm_file_path:
        raise ValueError(
            "captioning.llm.text and captioning.llm.file_path are mutually exclusive. "
            "Please specify only one."
        )

    # text/file_path captioners are standalone — combining with VLM is invalid
    if (llm_text or llm_file_path) and has_vlm:
        raise ValueError(
            "captioning.llm.text / file_path cannot be combined with captioning.vlm. "
            "Use VLM+LLM (without text/file_path) or text/file_path alone."
        )

    if llm_text:
        variables = (llm_config or {}).get("variables")
        return TextCaptioner(text=llm_text, logger=logger, variables=variables)

    if llm_file_path:
        seed = (llm_config or {}).get("seed")
        return FileCaptioner(file_path=llm_file_path, logger=logger, seed=seed)

    if has_vlm and has_llm:
        return _create_vlm_llm_captioner(
            vlm_config, llm_config, endpoints_dict, logger, config_dir
        )

    if has_vlm:
        return _create_vlm_captioner(vlm_config, endpoints_dict, logger, config_dir)

    if has_llm:
        return _create_llm_captioner(llm_config, endpoints_dict, logger, config_dir)

    raise ValueError("captioning section must contain at least one of: 'vlm', 'llm'")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_vlm_endpoint(endpoints: dict):
    """Resolve VLM endpoint URL and model from env vars or config."""
    vlm_ep = endpoints.get("vlm") or {}
    endpoint = os.getenv("VLM_ENDPOINT_URL") or vlm_ep.get("url", "")
    model = os.getenv("VLM_ENDPOINT_MODEL") or vlm_ep.get("model", "")
    if not endpoint:
        raise ValueError(
            "VLM captioning requires a VLM endpoint (endpoints.vlm.url or VLM_ENDPOINT_URL)"
        )
    return endpoint, model


def _resolve_llm_endpoint(endpoints: dict):
    """Resolve LLM endpoint URL and model from env vars or config."""
    llm_ep = endpoints.get("llm") or {}
    endpoint = (
        os.getenv("LLM_CAPTION_ENDPOINT_URL")
        or os.getenv("LLM_ENDPOINT_URL")
        or llm_ep.get("url", "")
    )
    model = (
        os.getenv("LLM_CAPTION_ENDPOINT_MODEL")
        or os.getenv("LLM_ENDPOINT_MODEL")
        or llm_ep.get("model", "")
    )
    if not endpoint:
        raise ValueError(
            "LLM captioning requires an LLM endpoint (endpoints.llm.url or LLM_ENDPOINT_URL)"
        )
    return endpoint, model


def _resolve_system_prompt(
    config_section: dict, config_dir: str = None, logger=None
) -> str:
    """Resolve system_prompt from inline text or system_prompt_file."""
    if config_section.get("system_prompt_file"):
        return load_system_prompt_from_config(config_section, config_dir, logger)
    return config_section.get("system_prompt", "")


def _create_vlm_captioner(
    vlm_config: dict, endpoints: dict, logger, config_dir: str = None
) -> VLMCaptioner:
    endpoint, model = _resolve_vlm_endpoint(endpoints)
    params = vlm_config.get("parameters") or {}
    return VLMCaptioner(
        system_prompt=_resolve_system_prompt(vlm_config, config_dir, logger),
        user_prompt=vlm_config.get("user_prompt", ""),
        retry=params.get("retry", 0),
        temperature=params.get("temperature", 0.3),
        top_p=params.get("top_p", 0.95),
        frequency_penalty=params.get("frequency_penalty", 1.05),
        max_tokens=params.get("max_tokens", 4096),
        stream=params.get("stream", False),
        parser=vlm_config.get("parser", "instruct"),
        fps=params.get("fps", 4.0),
        max_pixels=params.get("max_pixels", 307200),
        endpoint=endpoint,
        model=model,
        logger=logger,
    )


def _create_llm_captioner(
    llm_config: dict, endpoints: dict, logger, config_dir: str = None
) -> LLMCaptioner:
    endpoint, model = _resolve_llm_endpoint(endpoints)
    params = llm_config.get("parameters") or {}
    variables = llm_config.get("variables") or llm_config.get("verification_values")
    if not variables:
        raise ValueError(
            "LLM captioner requires 'captioning.llm.variables' or "
            "'captioning.llm.verification_values'"
        )
    return LLMCaptioner(
        variables=variables,
        system_prompt=_resolve_system_prompt(llm_config, config_dir, logger),
        example_prompt=llm_config.get("example_prompt", ""),
        retry=params.get("retry", 1),
        temperature=params.get("temperature", 0.3),
        top_p=params.get("top_p", 0.95),
        frequency_penalty=params.get("frequency_penalty", 1.05),
        presence_penalty=params.get("presence_penalty", 0.0),
        max_tokens=params.get("max_tokens", 512),
        stream=params.get("stream", True),
        endpoint=endpoint,
        model=model,
        logger=logger,
    )


def _create_vlm_llm_captioner(
    vlm_config: dict, llm_config: dict, endpoints: dict, logger, config_dir: str = None
) -> VLMLLMCaptioner:
    vlm_captioner = _create_vlm_captioner(vlm_config, endpoints, logger, config_dir)
    llm_captioner = _create_llm_captioner(llm_config, endpoints, logger, config_dir)
    return VLMLLMCaptioner(
        vlm_captioner=vlm_captioner,
        llm_captioner=llm_captioner,
        logger=logger,
    )
