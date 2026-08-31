# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Visual GenAI NIM adapter (REST ``/v1/infer``).

One adapter for every Visual GenAI NIM that speaks ``POST /v1/infer`` — image
NIMs (Qwen-Image-Edit) and video NIMs (Cosmos Transfer / Predict 2.5) alike. The
adapter imposes **no** body structure beyond what it must inject at runtime: the
config's ``parameters`` are forwarded verbatim (``body.update(params)``), so the
config alone constructs the payload, including arbitrarily nested fields such as
``parameters.edge.control_weight: 1.0`` -> ``{"edge": {"control_weight": 1.0}}``.

The only two things the adapter supplies itself (they can't live in YAML):
  * the **input media**, injected under a field name AND encoding chosen by
    media kind — the two real NIM contracts differ in both: video NIMs (Cosmos
    Transfer) take ``"video"`` as **raw base64**, while image NIMs
    (Qwen-Image-Edit) take ``"image"`` as a ``data:<mime>;base64,...`` URL;
  * **decoding the response**, accepting the common single-field variants
    (``b64_video`` for video NIMs, ``artifacts[0].base64`` / ``image`` / ... for
    image NIMs).

Reference::

    POST {url}/v1/infer
    image-edit: {"prompt", "image": "data:image/png;base64,..."} -> {"artifacts":[{"base64":...}]}
    cosmos:     {"prompt", "video": "<raw base64 mp4>", "seed", "guidance",
                 "num_steps", "sigma_max", "edge": {"control_weight": 1.0}, ...}
                                                                 -> {"b64_video": "..."}
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import quote, urlparse

import requests

from .base import BaseAdapter, Result, media_items, strip_url_credentials

# Default per-request timeout for NIM calls. Larger than the generic default
# because video models (Cosmos) can take minutes; image-edit returns fast.
NIM_DEFAULT_TIMEOUT = 300.0
NVCF_POLL_SECONDS_DEFAULT = 300
NVCF_POLL_SECONDS_MAX = 300
NVCF_STATUS_BASE_URL_DEFAULT = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status"


def _nvcf_status_base_url() -> str:
    return os.getenv("NVCF_STATUS_BASE_URL") or NVCF_STATUS_BASE_URL_DEFAULT


def _is_nvcf_url(url: str) -> bool:
    hostname = urlparse(url).hostname or ""
    return hostname == "nvcf.nvidia.com" or hostname.endswith(".nvcf.nvidia.com")


def _nvcf_poll_seconds() -> int:
    """Read the NVCF long-poll window, capped at the gateway maximum."""
    raw = os.getenv("NVCF_POLL_SECONDS", str(NVCF_POLL_SECONDS_DEFAULT))
    try:
        seconds = int(raw)
    except ValueError as exc:
        raise ValueError("NVCF_POLL_SECONDS must be an integer") from exc
    if seconds < 1:
        raise ValueError("NVCF_POLL_SECONDS must be at least 1")
    return min(seconds, NVCF_POLL_SECONDS_MAX)


class NIMAdapter(BaseAdapter):
    """Visual GenAI NIM contract: ``POST /v1/infer`` returning base64 artifacts."""

    INFER_ROUTE = "/v1/infer"

    def __init__(self, endpoint, logger):
        super().__init__(endpoint, logger)
        # Optional explicit override for the input-media field name; otherwise it
        # is chosen by media kind (video/* -> "video", else "image"). Read
        # defensively so a duck-typed endpoint may set it without a schema change.
        self.input_field = getattr(endpoint, "input_field", None)
        # Honor an explicit endpoint timeout; otherwise use the NIM default
        # rather than the short generic one.
        if getattr(endpoint, "timeout", None) is None:
            self.timeout = NIM_DEFAULT_TIMEOUT
        # The route already carries the ``/v1`` prefix, so tolerate an
        # OpenAI-style base URL (``.../v1``) and avoid a doubled ``/v1/v1``.
        base = self.url[:-3] if self.url.endswith("/v1") else self.url
        self.infer_url = f"{base}{self.INFER_ROUTE}"
        self.is_nvcf = _is_nvcf_url(self.infer_url)
        self.nvcf_poll_seconds = _nvcf_poll_seconds() if self.is_nvcf else None
        self.request_url = strip_url_credentials(self.infer_url)
        self.logger.info(
            "NIMAdapter ready: %s model=%s (timeout %.0fs)",
            self.infer_url,
            self.model,
            self.timeout,
        )

    def invoke(self, payload: Dict[str, Any]) -> Result:
        body: Dict[str, Any] = {}
        prompt = payload.get("prompt")
        if prompt:
            body["prompt"] = prompt
        body.update(payload.get("params") or {})
        if self.model:
            body.setdefault("model", self.model)

        for name, item in media_items(payload):
            b64 = base64.b64encode(item["bytes"]).decode("utf-8")
            mime = item.get("mime", "image/png")
            if item.get("kind") == "control":
                # Control modality (edge/depth/seg/vis) for Cosmos Transfer: nest
                # the read+encoded control video under its modality field, next to
                # any ``control_weight`` the config already placed there, e.g.
                #   parameters.edge.control_weight -> body["edge"]["control_weight"]
                #   + this control video           -> body["edge"]["control"]
                slot = body.get(name)
                if slot is None:
                    slot = {}
                    body[name] = slot
                elif not isinstance(slot, dict):
                    # params already set this modality to a non-object (e.g. a
                    # bool/number); attaching the control here would silently drop
                    # it. Fail loud so the config is fixed.
                    raise ValueError(
                        f"NIMAdapter: params.{name} must be an object "
                        f"(e.g. {{control_weight: ...}}) to attach a control input, "
                        f"got {type(slot).__name__}"
                    )
                slot["control"] = b64
            else:
                # Primary input media. The two real NIM contracts differ in BOTH
                # field name and encoding, keyed by media kind:
                #   * video NIMs (Cosmos Transfer): field "video", RAW base64 (no
                #     data: prefix) — confirmed against the NIM executor schema.
                #   * image NIMs (Qwen-Image-Edit): field "image", a data URL
                #     (``data:<mime>;base64,...``) — raw b64 is rejected with 422
                #     "Image has been provided in the invalid form".
                is_video = mime.startswith("video/")
                field = self.input_field or ("video" if is_video else "image")
                body[field] = b64 if is_video else f"data:{mime};base64,{b64}"

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.is_nvcf:
            headers["NVCF-POLL-SECONDS"] = str(
                min(self.nvcf_poll_seconds, max(1, int(self.timeout) - 1))
            )
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        deadline = time.monotonic() + self.timeout
        resp = requests.post(
            self.infer_url, json=body, headers=headers, timeout=self.timeout
        )
        nvcf_request_id = None
        if self.is_nvcf and resp.status_code == 202:
            resp, nvcf_request_id = self._poll_nvcf(resp, headers, deadline)
        if not resp.ok:
            # raise_for_status() drops the response body, but that body is where
            # the NIM puts the real reason (e.g. "UnsupportedVideoFormat: ...
            # 'yuvj420p' ...", a 422 field error, etc.). Surface it so the
            # pipeline log shows the cause without digging into the container.
            detail = (resp.text or "").strip()
            raise requests.HTTPError(
                f"NIM {self.infer_url} returned HTTP {resp.status_code}: "
                f"{detail[:2000] if detail else '<empty response body>'}",
                response=resp,
            )
        data = resp.json()

        media_bytes = _decode_artifact(data)
        if media_bytes is None:
            raise ValueError(
                f"NIMAdapter: no artifact in response from {self.infer_url} "
                f"(keys={list(data) if isinstance(data, dict) else type(data).__name__})"
            )
        # Recorded only now: a 200 carrying no artifact still fails the call, and
        # a call that produced nothing must not appear in the provenance record.
        self.record_request(body)
        request_id = nvcf_request_id or (
            data.get("id") if isinstance(data, dict) else None
        )
        return Result(media_bytes=media_bytes, request_id=request_id, raw=data)

    def _poll_nvcf(self, response, headers, deadline: float):
        """Long-poll an accepted hosted NVCF invocation until it completes."""
        request_id = response.headers.get("NVCF-REQID")
        if not request_id:
            raise ValueError("NVCF returned HTTP 202 without an NVCF-REQID")

        status_url = _nvcf_status_base_url().rstrip("/") + "/" + quote(
            str(request_id), safe=""
        )
        self.logger.info("NVCF request accepted request_id=%s; polling", request_id)
        poll_headers = {
            name: value for name, value in headers.items() if name != "Content-Type"
        }
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise requests.Timeout(
                    f"NVCF request {request_id} did not complete within "
                    f"{self.timeout:.0f}s"
                )
            poll_headers["NVCF-POLL-SECONDS"] = str(
                min(self.nvcf_poll_seconds, max(1, int(remaining) - 1))
            )
            response = requests.get(status_url, headers=poll_headers, timeout=remaining)
            if response.status_code != 202:
                self.logger.info(
                    "NVCF request finished request_id=%s status=%s",
                    request_id,
                    response.status_code,
                )
                return response, str(request_id)
            time.sleep(1)


def _decode_artifact(data: Any) -> Optional[bytes]:
    """Pull base64 media bytes out of a Visual GenAI NIM response.

    Handles the image-edit shape (``artifacts[0].base64``) and the common
    single-field variants used across NIMs, including ``b64_video`` for video
    models (Cosmos Transfer/Predict). The bytes are written as-is to the
    configured output path, so the file extension (``.mp4`` / ``.png``) decides
    the container — no per-type branching needed here.
    """
    if not isinstance(data, dict):
        return None
    artifacts = data.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        b64 = artifacts[0].get("base64") if isinstance(artifacts[0], dict) else None
        if b64:
            return base64.b64decode(b64)
    # Single-field variants across NIMs (video first, then image).
    for key in ("b64_video", "b64_json", "video", "image", "base64"):
        b64 = data.get(key)
        if isinstance(b64, str) and b64:
            return base64.b64decode(b64)
    return None
