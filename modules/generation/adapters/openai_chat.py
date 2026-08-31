# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI ``chat.completions`` adapter.

Serves VLM, LLM, hosted chat models (e.g. Gemma, Veo via inference-api),
Cosmos3 on vllm-omni, and image-edit deployments that speak chat.completions.
Generation params (``payload["params"]``) are passed straight into the request
body, so the config -- not the adapter -- decides their shape: a nested envelope
like ``extra_body:`` (Qwen image-edit) or flat top-level fields (standard chat).
"""

from __future__ import annotations

import base64
from typing import Any, Dict

from openai import OpenAI

from .base import BaseAdapter, Result, media_items, strip_url_credentials


class OpenAIChatAdapter(BaseAdapter):
    """OpenAI-compatible ``chat.completions`` contract (text- and image-out)."""

    @classmethod
    def for_chat(
        cls,
        url,
        model,
        logger,
        *,
        role=None,
        timeout=None,
        api_key_env=None,
    ):
        """Build an adapter from a bare URL/model (no hand-built ``Endpoint``).

        Convenience constructor for captioners/verifiers that hold a URL string
        rather than a Pydantic ``Endpoint``. ``api_key_env`` is threaded onto the
        inline endpoint so :func:`resolve_api_key` reads the key from that env var
        (``$api_key_env`` → role default). Schema is imported lazily to avoid
        import-order issues.
        """
        from aug_utils.schema.endpoints import Endpoint

        ep = Endpoint(
            id=f"{role or 'chat'}-inline",
            role=role or "llm",
            url=url,
            model=model or "",
            timeout=timeout,
            api_key_env=api_key_env,
        )
        return cls(ep, logger)

    def __init__(self, endpoint, logger):
        super().__init__(endpoint, logger)
        # max_retries=0: a wedged/slow endpoint should fail after `timeout`
        # rather than the SDK silently re-issuing; pipeline.retry is the single
        # source of retries.
        self.client = OpenAI(
            base_url=self.url,
            api_key=self.api_key or "dummy",
            timeout=self.timeout,
            max_retries=0,
        )
        # The SDK appends this route to base_url; state it so provenance reports
        # the same kind of value here as the adapters that build their own URL.
        self.request_url = strip_url_credentials(f"{self.url}/chat/completions")
        self.logger.info(
            "OpenAIChatAdapter ready: url=%s model=%s (timeout %.0fs, no SDK retries)",
            self.url,
            self.model,
            self.timeout,
        )

    # ------------------------------------------------------------------
    # Generation path (normalized payload in, normalized Result out)
    # ------------------------------------------------------------------
    def invoke(self, payload: Dict[str, Any]) -> Result:
        prompt = payload.get("prompt")
        params = payload.get("params") or {}

        content: list = []
        if prompt:
            content.append({"type": "text", "text": prompt})
        for _name, item in media_items(payload):
            b64 = base64.b64encode(item["bytes"]).decode("utf-8")
            mime = item.get("mime", "image/png")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        if params:
            # Pass the configured params straight into the request body (the SDK
            # merges extra_body's keys at the top level). The CONFIG expresses the
            # exact shape the target model wants -- a nested envelope such as
            # `extra_body:` / `extra_param:` (e.g. Qwen image-edit reads diffusion
            # knobs from a top-level `extra_body` object) OR flat top-level fields
            # (standard chat: temperature/top_p/...). The adapter makes NO envelope
            # assumption, so one adapter serves heterogeneous OpenAI-compatible
            # models. NOTE: a server silently ignores fields it doesn't recognize,
            # so the config must use the structure that model expects.
            kwargs["extra_body"] = dict(params)

        response = self.client.chat.completions.create(**kwargs)
        self.record_request(kwargs)
        request_id = getattr(response, "id", None)
        message = response.choices[0].message
        message_content = message.content

        # Text out (caption / LLM / VLM answer). A whitespace-only string with an
        # image alongside it (some gateways do this) is treated as image-out below.
        if isinstance(message_content, str) and message_content.strip():
            return Result(text=message_content, request_id=request_id, raw=response)

        # Image out. Two wire shapes are seen in the wild:
        #   1. content is a list whose first item carries an image_url data URL
        #      (Qwen image-edit on vllm-omni).
        #   2. content is null and the image lives in a sibling `images` array of
        #      {image_url:{url:data...}} (LiteLLM gateways fronting Gemini / "nano
        #      banana" on inference-api.nvidia.com, plus other vertex_ai models).
        media_bytes = _decode_image_content(message_content) or _decode_images_field(
            message
        )
        if media_bytes is None:
            raise ValueError(
                f"OpenAIChatAdapter: no text or image in response (request_id={request_id})"
            )
        return Result(media_bytes=media_bytes, request_id=request_id, raw=response)

    # ------------------------------------------------------------------
    # Captioning / verification path (full control over messages + sampling)
    # ------------------------------------------------------------------
    def chat(self, *, model: str | None = None, **kwargs):
        """Thin shared wrapper over ``chat.completions.create``.

        Lets captioners/verifiers reuse the one client (and its no-retry policy)
        while keeping their own message building, streaming, and parsing.
        Returns the raw SDK response (or stream iterator when ``stream=True``).

        Captioning and verification reach the endpoint through here rather than
        through :meth:`invoke`, so this is where their provenance is captured —
        the system and user prompts that produced the caption or the judgement.
        """
        resolved_model = model or self.model
        response = self.client.chat.completions.create(model=resolved_model, **kwargs)
        self.record_request({"model": resolved_model, **kwargs})
        return response


def _data_url_to_bytes(image_url) -> bytes | None:
    """Decode a ``data:<mime>;base64,<payload>`` URL to raw bytes."""
    if not image_url or "base64," not in image_url:
        return None
    return base64.b64decode(image_url.split("base64,", 1)[1])


def _item_image_url(item) -> str | None:
    """Read ``item.image_url.url`` from an SDK object or a plain dict."""
    try:
        return item.image_url.url
    except AttributeError:
        return item.get("image_url", {}).get("url") if isinstance(item, dict) else None


def _decode_image_content(content) -> bytes | None:
    """Extract image bytes from a chat ``message.content`` list (data URL).

    ``message.content`` is a list of typed parts in no guaranteed order, so scan
    every part: a leading text or refusal part must not hide an image carried in
    a later part.
    """
    if not content or isinstance(content, str):
        return None
    for item in content:
        image_bytes = _data_url_to_bytes(_item_image_url(item))
        if image_bytes is not None:
            return image_bytes
    return None


def _decode_images_field(message) -> bytes | None:
    """Extract image bytes from a ``message.images`` array (LiteLLM/Gemini).

    Gateways that front Gemini / "nano banana" return generated images in an
    ``images`` list of ``{"image_url": {"url": "data:image/png;base64,..."}}``,
    leaving ``message.content`` null.
    """
    images = getattr(message, "images", None)
    if not images:
        return None
    return _data_url_to_bytes(_item_image_url(images[0]))
