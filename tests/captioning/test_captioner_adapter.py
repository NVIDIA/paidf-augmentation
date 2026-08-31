# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for VLM/LLM captioners routed through ``OpenAIChatAdapter``.

These exercise the request shape sent to ``adapter.chat`` and the parsing of a
canned (offline) response. No network or live endpoint is involved: the shared
adapter's ``chat`` method is patched to return ``SimpleNamespace`` stand-ins for
the OpenAI SDK response objects.
"""

import logging
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from captioning.factory import _resolve_llm_endpoint, _resolve_vlm_endpoint
from captioning.llm_captioner import LLMCaptioner
from captioning.vlm_captioner import VLMCaptioner

LOGGER = logging.getLogger("test")

# A 1x1 PNG (valid header) is enough; the captioners only base64-encode the
# bytes, they never decode the image.
PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _nonstream_response(content, reasoning_content=None):
    """A non-streaming chat.completions response stand-in."""
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    return SimpleNamespace(id="resp", choices=[SimpleNamespace(message=message)])


@pytest.fixture
def png_path():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "frame.png")
        with open(path, "wb") as f:
            f.write(PNG_BYTES)
        yield path


# ---------------------------------------------------------------------------
# VLMCaptioner
# ---------------------------------------------------------------------------
class TestVLMCaptioner:
    def _make(self, parser):
        return VLMCaptioner(
            system_prompt="You are a captioner.",
            user_prompt="Describe this image.",
            retry=0,
            temperature=0.2,
            top_p=0.9,
            frequency_penalty=0.1,
            max_tokens=128,
            stream=False,
            endpoint="http://localhost:9/v1",
            model="some/vlm",
            parser=parser,
            logger=LOGGER,
        )

    def test_instruct_request_shape_and_parse(self, png_path):
        cap = self._make("instruct")
        with patch.object(
            cap.adapter, "chat", return_value=_nonstream_response("A red car.")
        ) as chat:
            caption = cap.get_caption(png_path)

        assert caption == "A red car."
        _, kwargs = chat.call_args
        # Sampling params preserved verbatim.
        assert kwargs["model"] == "some/vlm"
        assert kwargs["temperature"] == 0.2
        assert kwargs["top_p"] == 0.9
        assert kwargs["frequency_penalty"] == 0.1
        assert kwargs["max_tokens"] == 128
        assert kwargs["stream"] is False
        # instruct mode: system prompt first, then the user message carrying the
        # image then the prompt. The system prompt MUST reach the VLM.
        messages = kwargs["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are a captioner."}
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert content[1] == {"type": "text", "text": "Describe this image."}
        # instruct disables thinking via the nested chat_template_kwargs.
        assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_video_forwards_configured_extra_body_verbatim(self):
        # A VLM that needs its own video knobs (Qwen2.5-VL / NIM shape) gets them
        # from config, not from a model-name heuristic. Any vendor's key names work.
        qwen_body = {
            "media_io_kwargs": {"video": {"fps": 4.0}},
            "mm_processor_kwargs": {
                "videos_kwargs": {"min_pixels": 1568, "max_pixels": 307200}
            },
        }
        cap = VLMCaptioner(
            system_prompt="",
            user_prompt="Describe this video.",
            retry=0,
            temperature=0.2,
            top_p=0.9,
            frequency_penalty=0.1,
            max_tokens=128,
            stream=False,
            endpoint="http://localhost:9/v1",
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            parser="instruct",
            logger=LOGGER,
            extra_body=qwen_body,
        )
        with tempfile.TemporaryDirectory() as d:
            mp4 = os.path.join(d, "clip.mp4")
            with open(mp4, "wb") as f:
                f.write(b"\x00\x00\x00\x18ftypmp42")
            with patch.object(
                cap.adapter, "chat", return_value=_nonstream_response("A street.")
            ) as chat:
                cap.get_caption(mp4)

        _, kwargs = chat.call_args
        sent = kwargs["extra_body"]
        assert sent["media_io_kwargs"] == {"video": {"fps": 4.0}}
        assert sent["mm_processor_kwargs"]["videos_kwargs"]["max_pixels"] == 307200
        # deep-copied, so the caller's config dict is never mutated in place
        assert sent is not qwen_body

    def test_video_without_extra_body_sends_no_vendor_keys(self):
        # Default: nothing vendor-specific. Gemma 400s if Qwen's video kwargs appear.
        cap = VLMCaptioner(
            system_prompt="",
            user_prompt="Describe this video.",
            retry=0,
            temperature=0.2,
            top_p=0.9,
            frequency_penalty=0.1,
            max_tokens=128,
            stream=False,
            endpoint="http://localhost:9/v1",
            model="nvidia/google/gemma-4-31b-it",
            parser="instruct",
            logger=LOGGER,
        )
        with tempfile.TemporaryDirectory() as d:
            mp4 = os.path.join(d, "clip.mp4")
            with open(mp4, "wb") as f:
                f.write(b"\x00\x00\x00\x18ftypmp42")
            with patch.object(
                cap.adapter, "chat", return_value=_nonstream_response("A street.")
            ) as chat:
                cap.get_caption(mp4)

        sent = chat.call_args.kwargs["extra_body"]
        assert "media_io_kwargs" not in sent
        assert "mm_processor_kwargs" not in sent

    def test_instruct_without_system_prompt_omits_system_message(self, png_path):
        # An empty system prompt must not inject a blank system message.
        cap = VLMCaptioner(
            system_prompt="",
            user_prompt="Describe this image.",
            retry=0,
            temperature=0.2,
            top_p=0.9,
            frequency_penalty=0.1,
            max_tokens=128,
            stream=False,
            endpoint="http://localhost:9/v1",
            model="some/vlm",
            parser="instruct",
            logger=LOGGER,
        )
        with patch.object(
            cap.adapter, "chat", return_value=_nonstream_response("A red car.")
        ) as chat:
            cap.get_caption(png_path)

        _, kwargs = chat.call_args
        messages = kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_reasoning_request_shape_and_answer_parse(self, png_path):
        cap = self._make("reasoning")
        canned = "Some thinking...<answer>A blue truck.</answer> trailing"
        with patch.object(
            cap.adapter, "chat", return_value=_nonstream_response(canned)
        ) as chat:
            caption = cap.get_caption(png_path)

        # Only the <answer>...</answer> contents are extracted.
        assert caption == "A blue truck."
        _, kwargs = chat.call_args
        messages = kwargs["messages"]
        # reasoning mode: system prompt then user (text + image).
        assert messages[0] == {"role": "system", "content": "You are a captioner."}
        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert user_content[0] == {"type": "text", "text": "Describe this image."}
        assert user_content[1]["type"] == "image_url"

    def test_reasoning_missing_answer_raises(self, png_path):
        cap = self._make("reasoning")
        with patch.object(
            cap.adapter, "chat", return_value=_nonstream_response("no tags here")
        ):
            with pytest.raises(Exception, match="No caption found"):
                cap.get_caption(png_path)


# ---------------------------------------------------------------------------
# LLMCaptioner
# ---------------------------------------------------------------------------
class TestLLMCaptioner:
    def _make(self, stream=False):
        return LLMCaptioner(
            variables={"weather": "sunny", "vehicle": "truck"},
            system_prompt="You write prompts.",
            endpoint="http://localhost:9/v1",
            model="some/llm",
            logger=LOGGER,
            example_prompt="An example.",
            retry=0,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=256,
            stream=stream,
        )

    def test_variables_rendered_and_prompt_extracted(self):
        cap = self._make()
        canned = '{"prompt": "A truck under sunny skies."}'
        with patch.object(
            cap.adapter, "chat", return_value=_nonstream_response(canned)
        ) as chat:
            result = cap.get_caption("unused/path")

        assert result == "A truck under sunny skies."
        _, kwargs = chat.call_args
        assert kwargs["model"] == "some/llm"
        assert kwargs["temperature"] == 0.3
        assert kwargs["top_p"] == 0.95
        assert kwargs["max_tokens"] == 256
        assert kwargs["stream"] is False
        assert kwargs["presence_penalty"] == 0.0
        assert kwargs["frequency_penalty"] == 0.0
        # enable_thinking disabled via extra_body.
        assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        # System prompt + user content carrying the rendered attribute pairs.
        messages = kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You write prompts."}
        user_content = messages[1]["content"]
        assert "weather: sunny" in user_content
        assert "vehicle: truck" in user_content
        assert "An example." in user_content

    def test_caption_key_also_extracted(self):
        cap = self._make()
        canned = '```json\n{"caption": "Rainy night street."}\n```'
        with patch.object(
            cap.adapter, "chat", return_value=_nonstream_response(canned)
        ):
            assert cap.generate_prompt("unused") == "Rainy night street."


# ---------------------------------------------------------------------------
# API key resolution: api_key_env / literal api_key reach the adapter.
# ---------------------------------------------------------------------------
class TestCaptionerApiKeyResolution:
    """The endpoint's api_key_env / api_key must reach ``adapter.api_key``.

    Resolution precedence at the adapter: literal ``api_key`` → ``$api_key_env``
    → role default (``VLM_API_KEY`` / ``LLM_API_KEY``). We patch the OpenAI
    client so no network call happens; ``adapter.api_key`` is resolved in
    ``BaseAdapter.__init__`` before the client is built.
    """

    def test_vlm_uses_api_key_env(self, monkeypatch):
        # Role default would be VLM_API_KEY; ensure it is NOT what we read.
        monkeypatch.setenv("VLM_API_KEY", "role-default-vlm")
        monkeypatch.setenv("NVCF_API_KEY", "from-nvcf-env")
        with patch("generation.adapters.openai_chat.OpenAI"):
            cap = VLMCaptioner(
                system_prompt="s",
                user_prompt="u",
                retry=0,
                temperature=0.2,
                top_p=0.9,
                frequency_penalty=0.1,
                max_tokens=128,
                stream=False,
                endpoint="http://localhost:9/v1",
                model="some/vlm",
                parser="instruct",
                logger=LOGGER,
                api_key_env="NVCF_API_KEY",
            )
        assert cap.adapter.api_key == "from-nvcf-env"
        assert cap.adapter.api_key != "role-default-vlm"

    def test_vlm_defaults_to_role_env(self, monkeypatch):
        # No api_key_env / api_key configured -> falls back to role default.
        monkeypatch.setenv("VLM_API_KEY", "role-default-vlm")
        with patch("generation.adapters.openai_chat.OpenAI"):
            cap = VLMCaptioner(
                system_prompt="s",
                user_prompt="u",
                retry=0,
                temperature=0.2,
                top_p=0.9,
                frequency_penalty=0.1,
                max_tokens=128,
                stream=False,
                endpoint="http://localhost:9/v1",
                model="some/vlm",
                parser="instruct",
                logger=LOGGER,
            )
        assert cap.adapter.api_key == "role-default-vlm"

    def test_resolve_endpoints_return_key_fields(self, monkeypatch):
        # Ensure env URL overrides don't interfere with this resolution.
        for var in (
            "VLM_ENDPOINT_URL",
            "VLM_ENDPOINT_MODEL",
            "LLM_ENDPOINT_URL",
            "LLM_CAPTION_ENDPOINT_URL",
            "LLM_ENDPOINT_MODEL",
            "LLM_CAPTION_ENDPOINT_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)

        endpoints = [
            {
                "role": "vlm",
                "url": "http://vlm:9/v1",
                "model": "v/m",
                "api_key_env": "NVCF_API_KEY",
            },
            {
                "role": "llm",
                "url": "http://llm:9/v1",
                "model": "l/m",
                "api_key_env": "OTHER_KEY",
            },
        ]
        v_url, v_model, v_key_env = _resolve_vlm_endpoint(endpoints)
        assert (v_url, v_model, v_key_env) == (
            "http://vlm:9/v1",
            "v/m",
            "NVCF_API_KEY",
        )
        l_url, l_model, l_key_env = _resolve_llm_endpoint(endpoints)
        assert (l_url, l_model, l_key_env) == (
            "http://llm:9/v1",
            "l/m",
            "OTHER_KEY",
        )

    def test_llm_uses_api_key_env(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "role-default-llm")
        monkeypatch.setenv("NVCF_API_KEY", "from-nvcf-env")
        with patch("generation.adapters.openai_chat.OpenAI"):
            cap = LLMCaptioner(
                variables={"weather": "sunny"},
                system_prompt="s",
                endpoint="http://localhost:9/v1",
                model="some/llm",
                logger=LOGGER,
                api_key_env="NVCF_API_KEY",
            )
        assert cap.adapter.api_key == "from-nvcf-env"
        assert cap.adapter.api_key != "role-default-llm"
