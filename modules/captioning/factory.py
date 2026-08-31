# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import logging

from aug_utils.endpoint_registry import select_endpoint
from aug_utils.prompt_loader import load_system_prompt_from_config
from aug_utils.schema.captioning import TemplateCaptioningConfig
from captioning.base import BaseCaptioner
from captioning.template_captioner import TemplateCaptioner
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
    - Both ``vlm`` and ``template``     → TemplateCaptioner
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
        # Convert to dict for downstream constructors that still expect dicts
        captioning_dict = captioning_cfg.model_dump() if captioning_cfg else {}
        # Keep endpoints as the registry list (Endpoint objects); the resolver
        # handles both objects and dicts.
        endpoints = list(config.endpoints) if config.endpoints else []
    else:
        captioning_dict = config.get("captioning", {})
        endpoints = config.get("endpoints") or []

    vlm_config = captioning_dict.get("vlm")
    llm_config = captioning_dict.get("llm")
    template_config = captioning_dict.get("template")
    has_vlm = vlm_config is not None
    has_llm = llm_config is not None
    has_template = template_config is not None

    # Infer captioner type from which fields are set
    llm_text = (llm_config or {}).get("text")
    llm_file_path = (llm_config or {}).get("file_path")

    # text and file_path are mutually exclusive
    if llm_text and llm_file_path:
        raise ValueError(
            "captioning.llm.text and captioning.llm.file_path are mutually exclusive. "
            "Please specify only one."
        )

    if has_template and has_llm:
        raise ValueError(
            "captioning.template cannot be combined with captioning.llm. "
            "Use deterministic VLM-template prompting or LLM prompt generation, "
            "not both."
        )
    if has_template and not has_vlm:
        raise ValueError(
            "captioning.template requires captioning.vlm to describe the source media"
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

    if has_template:
        vlm_captioner = _create_vlm_captioner(vlm_config, endpoints, logger, config_dir)
        return _create_template_captioner(template_config, vlm_captioner, logger)

    if has_vlm and has_llm:
        return _create_vlm_llm_captioner(
            vlm_config, llm_config, endpoints, logger, config_dir
        )

    if has_vlm:
        return _create_vlm_captioner(vlm_config, endpoints, logger, config_dir)

    if has_llm:
        return _create_llm_captioner(llm_config, endpoints, logger, config_dir)

    raise ValueError(
        "captioning section must contain at least one of: 'vlm', 'llm', 'template'"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ep_attr(ep, name: str, default: str = "") -> str:
    """Read ``name`` from an endpoint object or dict, defaulting to ``default``."""
    if ep is None:
        return default
    if isinstance(ep, dict):
        return ep.get(name) or default
    return getattr(ep, name, None) or default


def _resolve_vlm_endpoint(endpoints: list, endpoint_id=None):
    """Resolve VLM endpoint URL, model, and key envs.

    Returns ``(url, model, api_key_env, api_key)``. URL/model honor env-var
    precedence; the key fields come from the resolved endpoint (``None`` when no
    endpoint resolves, e.g. an env-only URL).
    """
    ep = select_endpoint(endpoints, "vlm", endpoint_id, required=False)
    endpoint = os.getenv("VLM_ENDPOINT_URL") or _ep_attr(ep, "url")
    model = os.getenv("VLM_ENDPOINT_MODEL") or _ep_attr(ep, "model")
    api_key_env = _ep_attr(ep, "api_key_env", None)
    if not endpoint:
        raise ValueError(
            "VLM captioning requires a VLM endpoint (an endpoint with role "
            "'vlm', or VLM_ENDPOINT_URL)"
        )
    return endpoint, model, api_key_env


def _resolve_llm_endpoint(endpoints: list, endpoint_id=None):
    """Resolve LLM endpoint URL, model, and key envs.

    Returns ``(url, model, api_key_env, api_key)``. URL/model honor env-var
    precedence; the key fields come from the resolved endpoint (``None`` when no
    endpoint resolves, e.g. an env-only URL).
    """
    ep = select_endpoint(endpoints, "llm", endpoint_id, required=False)
    endpoint = (
        os.getenv("LLM_CAPTION_ENDPOINT_URL")
        or os.getenv("LLM_ENDPOINT_URL")
        or _ep_attr(ep, "url")
    )
    model = (
        os.getenv("LLM_CAPTION_ENDPOINT_MODEL")
        or os.getenv("LLM_ENDPOINT_MODEL")
        or _ep_attr(ep, "model")
    )
    api_key_env = _ep_attr(ep, "api_key_env", None)
    if not endpoint:
        raise ValueError(
            "LLM captioning requires an LLM endpoint (an endpoint with role "
            "'llm', or LLM_ENDPOINT_URL)"
        )
    return endpoint, model, api_key_env


def _resolve_system_prompt(
    config_section: dict, config_dir: str = None, logger=None
) -> str:
    """Resolve system_prompt from inline text or system_prompt_file."""
    if config_section.get("system_prompt_file"):
        return load_system_prompt_from_config(config_section, config_dir, logger)
    return config_section.get("system_prompt", "")


def _create_vlm_captioner(
    vlm_config: dict, endpoints: list, logger, config_dir: str = None
) -> VLMCaptioner:
    endpoint, model, api_key_env = _resolve_vlm_endpoint(
        endpoints, vlm_config.get("endpoint_id")
    )
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
        extra_body=params.get("extra_body"),
        endpoint=endpoint,
        model=model,
        logger=logger,
        api_key_env=api_key_env,
    )


def _create_llm_captioner(
    llm_config: dict, endpoints: list, logger, config_dir: str = None
) -> LLMCaptioner:
    endpoint, model, api_key_env = _resolve_llm_endpoint(
        endpoints, llm_config.get("endpoint_id")
    )
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
        api_key_env=api_key_env,
    )


def _create_vlm_llm_captioner(
    vlm_config: dict, llm_config: dict, endpoints: list, logger, config_dir: str = None
) -> VLMLLMCaptioner:
    vlm_captioner = _create_vlm_captioner(vlm_config, endpoints, logger, config_dir)
    llm_captioner = _create_llm_captioner(llm_config, endpoints, logger, config_dir)
    return VLMLLMCaptioner(
        vlm_captioner=vlm_captioner,
        llm_captioner=llm_captioner,
        logger=logger,
    )


def _create_template_captioner(
    template_config: dict,
    vlm_captioner: VLMCaptioner,
    logger,
) -> TemplateCaptioner:
    """Create a deterministic renderer around the existing VLM client."""
    validated = TemplateCaptioningConfig.model_validate(template_config)
    return TemplateCaptioner(
        template=validated.template,
        attributes=validated.attributes.model_dump(),
        vlm_captioner=vlm_captioner,
        logger=logger,
    )
