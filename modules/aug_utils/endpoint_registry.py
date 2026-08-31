# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure config-resolution helpers for the BYOM multi-endpoint generation layer.

These helpers operate on a *list* of endpoint objects. Each endpoint is
duck-typed — it may be an object exposing the attributes ``id``, ``role``,
``url``, ``model``, ``adapter`` (e.g. the Pydantic ``Endpoint`` model), or a
plain ``dict`` with those keys. No transport or schema imports here; this module
only maps selectors/roles to endpoints and to default adapter contracts.
"""

from __future__ import annotations

from typing import Any, List, Optional

# augmentation.model.name -> endpoint role.
ROLE_FOR_MODEL_NAME = {
    "image-edit": "image_edit",
    "cosmos-transfer2.5": "video_transfer",
    "cosmos-predict": "video_predict",
    "cosmos3-image2video": "image2video",
}

# Default API-contract adapter for each role when an endpoint declares none.
DEFAULT_ADAPTER_FOR_ROLE = {
    "vlm": "openai.chat.completions",
    "llm": "openai.chat.completions",
    "image_edit": "nim",
    "video_transfer": "nim",
    "video_predict": "nim",
    # image-to-video generators (Cosmos3) speak the synchronous OpenAI video
    # contract by default; an async endpoint (Veo) sets adapter: openai.video.async.
    "image2video": "openai.video.sync",
}


def _attr(endpoint: Any, name: str) -> Any:
    """Read ``name`` from an endpoint given as an object or a dict."""
    if isinstance(endpoint, dict):
        return endpoint.get(name)
    return getattr(endpoint, name, None)


def default_adapter_for(role: Optional[str]) -> Optional[str]:
    """Return the default adapter contract for ``role`` (``None`` if unknown)."""
    return DEFAULT_ADAPTER_FOR_ROLE.get(role)


def find_endpoint_by_role(
    endpoints: List[Any], role: str, required: bool = True
) -> Any:
    """Return the single endpoint whose ``role`` matches.

    Raises ``ValueError`` if more than one endpoint has that role. When zero
    match: raises if ``required`` (default), else returns ``None``.
    """
    matches = [ep for ep in endpoints if _attr(ep, "role") == role]
    if not matches:
        if required:
            raise ValueError(f"no endpoint found with role {role!r}")
        return None
    if len(matches) > 1:
        ids = [_attr(ep, "id") for ep in matches]
        raise ValueError(
            f"multiple endpoints found with role {role!r}: {ids}; disambiguate by id"
        )
    return matches[0]


def select_endpoint(
    endpoints: List[Any],
    role: str,
    endpoint_id: Optional[str] = None,
    *,
    required: bool = True,
) -> Any:
    """Resolve the endpoint for a consumer of ``role``.

    With ``endpoint_id``: return that endpoint, raising if it is absent or its
    role does not match ``role``. Without: fall back to the single endpoint of
    ``role`` (:func:`find_endpoint_by_role`).
    """
    if endpoint_id is not None:
        for ep in endpoints:
            if _attr(ep, "id") == endpoint_id:
                ep_role = _attr(ep, "role")
                if ep_role != role:
                    raise ValueError(
                        f"endpoint_id {endpoint_id!r} has role {ep_role!r}, but a "
                        f"{role!r} endpoint is required here"
                    )
                return ep
        available = [i for i in (_attr(e, "id") for e in endpoints) if i]
        raise ValueError(
            f"endpoint_id {endpoint_id!r} not found in endpoints; available ids: {available}"
        )
    return find_endpoint_by_role(endpoints, role, required=required)


def resolve_endpoint(endpoints: List[Any], selector: str) -> Any:
    """Resolve an endpoint from a string ``selector``.

    Resolution order: ``id == selector``, then ``role == selector``, then
    ``role == ROLE_FOR_MODEL_NAME.get(selector)``.

    Raises ``ValueError`` if nothing matches (listing available ids) or if a
    role match is ambiguous (asking to disambiguate by id).
    """
    # 1) Exact id match (ids are expected to be unique).
    for ep in endpoints:
        if _attr(ep, "id") == selector:
            return ep

    # 2) Direct role match, then 3) model-name -> role mapping.
    for role in (selector, ROLE_FOR_MODEL_NAME.get(selector)):
        if role is None:
            continue
        matches = [ep for ep in endpoints if _attr(ep, "role") == role]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            ids = [_attr(ep, "id") for ep in matches]
            raise ValueError(
                f"selector {selector!r} matches multiple endpoints with role "
                f"{role!r}: {ids}; disambiguate by id"
            )

    available = [_attr(ep, "id") for ep in endpoints]
    raise ValueError(
        f"no endpoint matched selector {selector!r}; available ids: {available}"
    )


def effective_adapter(endpoint: Any) -> str:
    """Return the adapter contract for ``endpoint``.

    Uses ``endpoint.adapter`` when set, else the role default. Raises
    ``ValueError`` if neither yields a contract.
    """
    adapter = _attr(endpoint, "adapter")
    if adapter:
        return adapter
    role = _attr(endpoint, "role")
    default = default_adapter_for(role)
    if default:
        return default
    raise ValueError(
        f"cannot determine adapter for endpoint "
        f"{_attr(endpoint, 'id')!r} (role={role!r})"
    )
