# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import tempfile
import os
import re
import base64
import shutil
import types

import multistorageclient as msc
from openai import OpenAI

from captioning.base import BaseCaptioner
from aug_utils.common import validate_and_cast_config_params
from aug_utils.constants import is_image


class VLMCaptioner(BaseCaptioner):
    """Vision-Language Model based captioning."""

    # Define expected parameter types for validation
    _PARAM_TYPES = {
        "retry": int,
        "temperature": float,
        "top_p": float,
        "frequency_penalty": float,
        "max_tokens": int,
        "stream": bool,
        "system_prompt": str,
        "user_prompt": str,
        "endpoint": str,
        "model": str,
        "parser": str,
        "fps": float,
        "max_pixels": int,
    }

    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        retry: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        max_tokens: int,
        stream: bool,
        endpoint: str,
        model: str,
        parser: str,
        fps: float,
        max_pixels: int,
        logger: logging.Logger,
    ):
        super().__init__(logger)

        # Validate and cast parameters to ensure correct types
        params = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "retry": retry,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "stream": stream,
            "endpoint": endpoint,
            "model": model,
            "parser": parser,
            "fps": fps,
            "max_pixels": max_pixels,
        }

        validated_params = validate_and_cast_config_params(
            params, self._PARAM_TYPES, logger
        )

        self.system_prompt = validated_params["system_prompt"]
        self.user_prompt = validated_params["user_prompt"]

        self.retry = validated_params["retry"]
        self.temperature = validated_params["temperature"]
        self.top_p = validated_params["top_p"]
        self.frequency_penalty = validated_params["frequency_penalty"]
        self.max_tokens = validated_params["max_tokens"]
        self.stream = validated_params["stream"]

        # Ensure endpoint URL is properly formatted (remove trailing slash to avoid double slashes)
        self.endpoint = validated_params["endpoint"].rstrip("/")
        self.model = validated_params["model"]
        self.parser = validated_params["parser"]
        self.fps = validated_params["fps"]
        self.max_pixels = validated_params["max_pixels"]

        if self.parser not in ("reasoning", "instruct"):
            raise ValueError(
                f"Invalid parser mode: {self.parser}. Must be 'reasoning' or 'instruct'"
            )

    @staticmethod
    def _clean_caption(caption: str) -> str:
        caption = caption.replace("*", "")
        caption = caption.replace("#", "")
        caption = caption.replace("  -", "")
        caption = caption.replace("- ", "")
        caption = re.sub(r"^\d+\.\s*", "", caption, flags=re.MULTILINE)
        caption = caption.replace("  ", " ")
        caption = caption.replace("\n\n", "\n")
        return caption.strip()

    def get_caption(self, media_path: str) -> str:
        """
        Get caption for an image or video file using VLM.

        Args:
            media_path: Path to the image or video file

        Returns:
            str: Generated caption text

        Raises:
            Exception: If caption generation fails
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            if not msc.is_file(media_path):
                raise Exception(f"Media file not found: {media_path}")

            # Detect media type
            is_image_file = is_image(media_path)

            # Download media from storage to local temp
            if is_image_file:
                # Preserve original image extension
                file_ext = os.path.splitext(media_path)[1]
                temp_media_path = os.path.join(temp_dir, f"image{file_ext}")
            else:
                temp_media_path = os.path.join(temp_dir, "video.mp4")

            with msc.open(media_path, "rb") as src, open(temp_media_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            self.logger.debug(f"Downloaded media to {temp_media_path}")

            try:
                api_key = os.environ.get("VLM_API_KEY")
                if not api_key:
                    self.logger.warning(
                        "No VLM_API_KEY set. Using 'not-used' as placeholder."
                    )
                    api_key = "not-used"

                client = OpenAI(base_url=self.endpoint, api_key=api_key, timeout=7200)

                # Encode media as base64 for NIM VLM API
                with open(temp_media_path, "rb") as media_file:
                    media_b64 = base64.b64encode(media_file.read()).decode()

                # Build content based on media type
                if is_image_file:
                    content_item = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{media_b64}"},
                    }
                    extra_params = {}
                    self.logger.debug(
                        "Sending chat completion request with image (parser: %s, model: %s)",
                        self.parser,
                        self.model,
                    )
                else:
                    content_item = {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{media_b64}"},
                    }
                    self.logger.debug(
                        "Sending chat completion request with video (parser: %s, model: %s)",
                        self.parser,
                        self.model,
                    )

                    is_qwen3_model = "qwen3" in self.model.lower()
                    if not is_qwen3_model:
                        extra_params = {
                            "extra_body": {
                                "media_io_kwargs": {"video": {"fps": self.fps}},
                                "mm_processor_kwargs": {
                                    "videos_kwargs": {
                                        "min_pixels": 1568,
                                        "max_pixels": self.max_pixels,
                                    }
                                },
                            }
                        }
                    else:
                        extra_params = {}
                        self.logger.debug(
                            "Skipping NIM extra_body params for Qwen3 model: %s",
                            self.model,
                        )

                # Disable thinking mode for instruct parser
                if self.parser == "instruct":
                    extra_body = extra_params.setdefault("extra_body", {})
                    chat_template_kwargs = extra_body.setdefault(
                        "chat_template_kwargs", {}
                    )
                    chat_template_kwargs.setdefault("enable_thinking", False)

                if self.parser == "reasoning":
                    conversation = [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.user_prompt},
                                content_item,
                            ],
                        },
                    ]
                else:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                content_item,
                                {"type": "text", "text": self.user_prompt},
                            ],
                        }
                    ]

                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=conversation,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    max_tokens=self.max_tokens,
                    stream=self.stream,
                    **extra_params,
                )

                if self.stream:
                    content_parts = []
                    reasoning_parts = []
                    for chunk in chat_response:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta.content is not None:
                            content_parts.append(delta.content)
                        reasoning_delta = getattr(delta, "reasoning_content", None)
                        if reasoning_delta is not None:
                            reasoning_parts.append(reasoning_delta)
                    content_text = "".join(content_parts)
                    if not content_text.strip() and reasoning_parts:
                        content_text = "".join(reasoning_parts)
                    assistant_message = types.SimpleNamespace(content=content_text)
                else:
                    if not chat_response.choices or len(chat_response.choices) == 0:
                        self.logger.warning(
                            "Chat response has no choices; using empty content"
                        )
                        assistant_message = types.SimpleNamespace(content="")
                    else:
                        assistant_message = chat_response.choices[0].message
                        if not (
                            assistant_message.content
                            and assistant_message.content.strip()
                        ):
                            reasoning_text = getattr(
                                assistant_message, "reasoning_content", None
                            )
                            if reasoning_text and reasoning_text.strip():
                                self.logger.warning(
                                    "Empty content; falling back to reasoning_content. "
                                    "Endpoint may have ignored enable_thinking=false."
                                )
                                assistant_message = types.SimpleNamespace(
                                    content=reasoning_text
                                )

                self.logger.debug(assistant_message.content)

                if self.parser == "reasoning":
                    match = re.search(
                        r"<answer>(.*?)</answer>", assistant_message.content, re.DOTALL
                    )
                    if match:
                        caption = match.group(1).strip()
                    else:
                        self.logger.error(
                            "No <answer> tags found in reasoning response: %s",
                            assistant_message.content,
                        )
                        raise Exception(
                            "No caption found in the response, please check if the parser "
                            "(reasoning or instruct) is correct"
                        )
                else:
                    if assistant_message.content and assistant_message.content.strip():
                        caption = assistant_message.content.strip()
                        caption = re.sub(
                            r"<think>.*?</think>", "", caption, flags=re.DOTALL
                        ).strip()
                    else:
                        self.logger.error(
                            "Empty response from instruct model: %s",
                            assistant_message.content,
                        )
                        raise Exception("No caption found in the response")

                caption = self._clean_caption(caption)
                if not caption:
                    raise Exception("Caption is empty after processing")

                self.logger.debug(f"Generated caption: {caption}")
                return caption

            except Exception as e:
                self.logger.error(f"Failed to get caption: {e}", exc_info=True)
                if self.retry > 0:
                    self.logger.info(f"Retrying {self.retry - 1} times")
                    return self._get_caption_retry(media_path, self.retry - 1)
                else:
                    raise e

    def _get_caption_retry(self, media_path: str, retries_left: int) -> str:
        """Internal retry method for caption generation."""
        original_retry = self.retry
        self.retry = retries_left
        try:
            return self.get_caption(media_path)
        finally:
            self.retry = original_retry
