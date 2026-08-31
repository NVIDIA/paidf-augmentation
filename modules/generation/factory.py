# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry-driven generator factory for the BYOM generation layer.

``create_generator`` resolves the configured ``augmentation.model.name`` to an
endpoint in the endpoints list, picks the API-contract adapter for that
endpoint, and wraps it in an executor. The result is a :class:`BaseGenerator`
(actually a :class:`BaseExecutor`) so the CLI invariant — ``execute(...)`` +
mutable ``.seed`` — keeps working unchanged.
"""

import logging
import time
from typing import Optional

from aug_utils.endpoint_registry import effective_adapter, resolve_endpoint
from aug_utils.schema.endpoints import Endpoint

from .adapters import (
    NIMAdapter,
    OpenAIChatAdapter,
    OpenAIImageEditAdapter,
    OpenAIVideoAsyncAdapter,
    OpenAIVideoSyncAdapter,
    PassthroughAdapter,
)
from .base import BaseGenerator
from .executors import BaseExecutor, seed_holder
from .inputs_spec import inputs_spec_for

# API-contract adapters, keyed by contract name. Mirrors
# ``aug_utils.schema.adapters.KNOWN_ADAPTERS``.
ADAPTERS = {
    "openai.chat.completions": OpenAIChatAdapter,
    "openai.images.edits": OpenAIImageEditAdapter,
    "openai.video.sync": OpenAIVideoSyncAdapter,
    "openai.video.async": OpenAIVideoAsyncAdapter,
    "nim": NIMAdapter,
    "passthrough": PassthroughAdapter,
}

# Per-role executor overrides. Empty for now; a future Cosmos executor that
# manages a local inference server registers here by role.
EXECUTOR_OVERRIDES = {}


def create_generator(config, logger: logging.Logger) -> BaseGenerator:
    """Create a generator from the ``augmentation`` configuration.

    Resolves ``augmentation.model.name`` against the endpoints list, builds the
    matching adapter, and returns a :class:`BaseExecutor` for the endpoint's
    role.

    Args:
        config: Full pipeline config (PipelineConfig model or raw dict).
        logger: Logger instance.

    Returns:
        BaseGenerator: Configured executor instance.

    Raises:
        ValueError: If augmentation is missing, no endpoint matches, or the
            resolved adapter contract is unknown.
    """
    augmentation, raw_endpoints = _read_sections(config)

    # Normalize endpoints to Endpoint objects so adapters get attribute access.
    endpoints = [
        ep if not isinstance(ep, dict) else Endpoint(**dict(ep))
        for ep in _as_list(raw_endpoints)
    ]

    name = (
        augmentation.model.name.value
        if hasattr(augmentation.model.name, "value")
        else str(augmentation.model.name)
    )

    ep = resolve_endpoint(endpoints, name)

    contract = effective_adapter(ep)
    adapter_cls = ADAPTERS.get(contract)
    if adapter_cls is None:
        raise ValueError(
            f"endpoint {ep.id!r} resolves to unknown adapter {contract!r}; "
            f"known adapters: {list(ADAPTERS)}"
        )
    # pipeline.request_timeout, when the user set it explicitly, overrides the
    # adapter's own default (image-edit 120s, video 900s, ...) -- but never a
    # per-endpoint timeout, which stays authoritative. An unset request_timeout
    # leaves each adapter on its default.
    request_timeout = _explicit_request_timeout(config)
    if request_timeout is not None and getattr(ep, "timeout", None) is None:
        ep = ep.model_copy(update={"timeout": request_timeout})
    adapter = adapter_cls(ep, logger)

    # Build params: only the knobs the user actually set (exclude_unset), so the
    # typed schema's defaults for OTHER models (e.g. cosmos sigma/num_steps) do
    # not leak onto the wire and get rejected by a strict endpoint. This is the
    # "pass through exactly what's configured" half of the hybrid param model.
    params = _dump(augmentation.parameters, exclude_unset=True)
    modalities = (
        _dump(augmentation.modalities, exclude_unset=True)
        if augmentation.modalities
        else {}
    )
    # Control-modality weights (edge/depth/seg/vis) must reach the adapter as
    # {control_weight: value} objects so a control input can attach to them; a
    # bare float is rejected by NIMAdapter ("params.<modality> must be an
    # object"). Prompt/string modality fields pass through unchanged.
    for _mod in ("edge", "depth", "seg", "vis"):
        weight = modalities.get(_mod)
        if weight is not None and not isinstance(weight, dict):
            modalities[_mod] = {"control_weight": weight}
    params.update(modalities)
    # Resolve seed where the config placed it (top-level or inside an envelope
    # such as `extra_body`), so retry re-roll mutates the seed the model reads.
    holder = seed_holder(params)
    holder["seed"] = _resolve_seed(holder.get("seed"), logger)

    inference_type = params.get("inference_type")
    if hasattr(inference_type, "value"):
        inference_type = inference_type.value
        params["inference_type"] = inference_type

    inputs_spec = inputs_spec_for(ep.role, inference_type)

    executor_cls = EXECUTOR_OVERRIDES.get(ep.role, BaseExecutor)
    return executor_cls(
        adapter=adapter, params=params, inputs_spec=inputs_spec, logger=logger
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_sections(config):
    """Return ``(augmentation, raw_endpoints)`` from a model or raw dict.

    The augmentation section is always returned as a Pydantic-like object
    exposing ``.model`` / ``.parameters`` / ``.modalities``; raw-dict configs
    are coerced into the typed model.
    """
    if hasattr(config, "augmentation"):
        aug = config.augmentation
        endpoints = config.endpoints
    else:
        aug = config.get("augmentation")
        endpoints = config.get("endpoints", [])

    if aug is None:
        raise ValueError("'augmentation' section is required to create a generator")

    if not hasattr(aug, "model"):
        from aug_utils.schema.augmentation import AugmentationConfig

        aug = AugmentationConfig(**dict(aug))

    return aug, endpoints


def _as_list(value):
    """Coerce an endpoints container (incl. OmegaConf) into a plain list."""
    if value is None:
        return []
    return list(value)


def _explicit_request_timeout(config) -> Optional[float]:
    """Return ``pipeline.request_timeout`` only when the user set it explicitly.

    Used as a generation-adapter timeout override. Returns ``None`` (keep the
    adapter's own default) when the field was left unset, so a defaulted
    ``request_timeout`` never shortens an adapter whose default is longer
    (e.g. the 900s video adapters).
    """
    if hasattr(config, "pipeline"):
        pipeline = config.pipeline
        if pipeline is not None and "request_timeout" in getattr(
            pipeline, "model_fields_set", set()
        ):
            return float(pipeline.request_timeout)
        return None
    pipeline = config.get("pipeline") if hasattr(config, "get") else None
    if isinstance(pipeline, dict) and pipeline.get("request_timeout") is not None:
        return float(pipeline["request_timeout"])
    return None


def _dump(obj, *, exclude_unset: bool = False) -> dict:
    """Return a plain dict for a Pydantic model or a mapping.

    ``exclude_unset`` drops fields left at their schema default so only
    explicitly-configured params reach the endpoint payload.
    """
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_unset=exclude_unset)
    return dict(obj)


def _resolve_seed(seed_value, logger: logging.Logger) -> int:
    """Resolve a seed value, generating one from time if None."""
    if seed_value is None or seed_value == "None" or seed_value == "none":
        # Sub-second entropy so two generators created within the same second
        # don't collide on int(time.time()); masked to 31 bits to stay in the
        # non-negative int32 range endpoints accept for a seed.
        seed = time.time_ns() % (2**31)
        logger.debug(f"Generated random seed: {seed}")
        return seed
    return int(seed_value)
