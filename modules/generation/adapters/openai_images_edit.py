# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Images *edits* adapter (``POST /v1/images/edits``).

For image-edit servers that expose the OpenAI **Images** API (the
``client.images.edit`` method) rather than ``chat.completions`` or the NIM
``/v1/infer`` route. The Qwen-Image-Edit NIM, for example, serves this route in
addition to ``/v1/infer``.

Wire shape::

    POST {url}/images/edits        (multipart/form-data)
    image=<file>, prompt="...", [size, n, response_format], extra_body={...}
        -> {"data": [{"b64_json": "..."}]}   (or {"url": "..."})

The input image is uploaded as multipart form-data (the SDK handles this); the
edited image returns as ``data[0].b64_json`` (default) or a URL we then fetch.
Contract-specific knobs (steps, cfg_scale, seed, ...) ride through ``extra_body``.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from openai import OpenAI

from .base import (
    BaseAdapter,
    Result,
    first_media,
    reject_control_media,
    strip_url_credentials,
)


def _same_origin(a: str, b: str) -> bool:
    """True when two URLs share scheme, host, and port (same origin)."""
    pa, pb = urlparse(a), urlparse(b)
    return (pa.scheme, pa.hostname, pa.port) == (pb.scheme, pb.hostname, pb.port)


# Named kwargs the OpenAI ``images.edit`` method accepts directly. Anything else
# in ``params`` is contract-specific and forwarded via ``extra_body``.
NATIVE_PARAMS = ("n", "size", "response_format", "user", "background", "quality")


class OpenAIImageEditAdapter(BaseAdapter):
    """OpenAI Images *edits* contract: ``client.images.edit`` over ``/v1/images/edits``."""

    def __init__(self, endpoint, logger):
        super().__init__(endpoint, logger)
        # max_retries=0: a wedged/slow endpoint fails after `timeout` rather than
        # the SDK silently re-issuing; pipeline.retry is the single retry source.
        self.client = OpenAI(
            base_url=self.url,
            api_key=self.api_key or "dummy",
            timeout=self.timeout,
            max_retries=0,
        )
        # The SDK appends this route to base_url; state it so provenance reports
        # the same kind of value here as the adapters that build their own URL.
        self.request_url = strip_url_credentials(f"{self.url}/images/edits")
        self.logger.info(
            "OpenAIImageEditAdapter ready: url=%s model=%s (timeout %.0fs, no SDK retries)",
            self.url,
            self.model,
            self.timeout,
        )

    def invoke(self, payload: Dict[str, Any]) -> Result:
        reject_control_media(payload, "OpenAIImageEditAdapter")
        media = first_media(payload)
        if media is None:
            raise ValueError("OpenAIImageEditAdapter: an input image is required")

        params = dict(payload.get("params") or {})
        # Split params into the SDK's native kwargs vs. extra (server-specific).
        native = {k: params.pop(k) for k in list(params) if k in NATIVE_PARAMS}
        # Ask for base64 by default so we get bytes back without a 2nd round-trip.
        native.setdefault("response_format", "b64_json")

        mime = media.get("mime", "image/png")
        ext = mime.split("/")[-1].split("+")[0] or "png"
        # SDK file form: (filename, bytes, content_type) -> multipart upload.
        image_file = (f"image.{ext}", media["bytes"], mime)

        kwargs: Dict[str, Any] = {
            "image": image_file,
            "prompt": payload.get("prompt") or "",
        }
        if self.model:
            kwargs["model"] = self.model
        kwargs.update(native)
        if params:
            kwargs["extra_body"] = dict(params)

        response = self.client.images.edit(**kwargs)
        self.record_request(kwargs)
        request_id = getattr(response, "_request_id", None) or getattr(
            response, "id", None
        )
        media_bytes = self._decode(response)
        if media_bytes is None:
            raise ValueError(
                f"OpenAIImageEditAdapter: no image in response (request_id={request_id})"
            )
        return Result(media_bytes=media_bytes, request_id=request_id, raw=response)

    def _decode(self, response) -> Optional[bytes]:
        """Pull image bytes from an Images API response (b64 inline, else URL)."""
        data = getattr(response, "data", None)
        if not data:
            return None
        item = data[0]
        b64 = getattr(item, "b64_json", None) or (
            item.get("b64_json") if isinstance(item, dict) else None
        )
        if b64:
            return base64.b64decode(b64)
        url = getattr(item, "url", None) or (
            item.get("url") if isinstance(item, dict) else None
        )
        if url:
            # Only fetch the result from the endpoint's own origin. A BYOM endpoint
            # (trusted only as far as the operator configured it) could otherwise
            # return a URL on an attacker-controlled host, turning this into an
            # SSRF / key-exfiltration primitive. Off-origin URLs are refused; the
            # default response_format=b64_json avoids the URL path entirely.
            if not _same_origin(url, self.url):
                raise ValueError(
                    f"OpenAIImageEditAdapter: refusing to fetch off-origin image "
                    f"URL {url} (endpoint origin {self.url}). Request b64_json "
                    f"output or point at a same-origin endpoint."
                )
            headers = (
                {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
            r = requests.get(url, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return r.content
        return None
