#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import re
from typing import Any, Dict

from openai import OpenAI

from captioning.base import BaseCaptioner
from aug_utils.common import validate_and_cast_config_params


class LLMCaptioner(BaseCaptioner):
    """LLM-based prompt generation from configured attribute values."""

    _PARAM_TYPES = {
        "system_prompt": str,
        "example_prompt": str,
        "retry": int,
        "temperature": float,
        "top_p": float,
        "frequency_penalty": float,
        "presence_penalty": float,
        "max_tokens": int,
        "stream": bool,
        "endpoint": str,
        "model": str,
    }

    def __init__(
        self,
        variables: Dict[str, Any],
        system_prompt: str,
        endpoint: str,
        model: str,
        logger: logging.Logger,
        example_prompt: str = "",
        retry: int = 1,
        temperature: float = 0.3,
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 512,
        stream: bool = True,
    ):
        super().__init__(logger)

        params = {
            "system_prompt": system_prompt,
            "example_prompt": example_prompt,
            "retry": retry,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "stream": stream,
            "endpoint": endpoint,
            "model": model,
        }
        validated = validate_and_cast_config_params(params, self._PARAM_TYPES, logger)

        self.variables = self._normalize_variables(variables)
        if not self.variables:
            raise ValueError("LLM captioner requires at least one variable value")

        self.system_prompt = validated["system_prompt"]
        self.example_prompt = validated["example_prompt"]
        self.retry = validated["retry"]
        self.temperature = validated["temperature"]
        self.top_p = validated["top_p"]
        self.frequency_penalty = validated["frequency_penalty"]
        self.presence_penalty = validated["presence_penalty"]
        self.max_tokens = validated["max_tokens"]
        self.stream = validated["stream"]
        self.endpoint = validated["endpoint"].rstrip("/")
        self.model = validated["model"]

        api_key = os.environ.get("LLM_API_KEY") or "not-used"
        self.client = OpenAI(base_url=self.endpoint, api_key=api_key)

    @staticmethod
    def _normalize_variables(variables: Dict[str, Any]) -> Dict[str, str]:
        normalized = {}
        for key, value in (variables or {}).items():
            if isinstance(value, list):
                if not value:
                    continue
                normalized[key] = str(value[0])
            elif value is not None:
                normalized[key] = str(value)
        return normalized

    def _build_user_content(self, source_caption: str = "") -> str:
        user_content = ""
        if source_caption:
            user_content += f"Source media caption:\n{source_caption.strip()}\n\n"
        user_content += "Attribute-value pairs:\n" + "\n".join(
            f"  {k}: {v}" for k, v in sorted(self.variables.items())
        )
        if self.example_prompt:
            user_content += (
                f"\n\nExample prompt (for style reference):\n{self.example_prompt}"
            )
        user_content += (
            "\n\nGenerate a single sentence prompt from the attribute-value pairs above. "
            'Output a JSON object with one key: "prompt" (or "caption") containing the '
            "sentence. No other text."
        )
        return user_content

    @staticmethod
    def _extract_prompt(text: str) -> str:
        cleaned = text.strip()

        if "```" in cleaned:
            code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
            if code_match:
                cleaned = code_match.group(1).strip()

        for match in re.finditer(r"\{[\s\S]*?\}", cleaned):
            try:
                obj = json.loads(match.group(0))
                prompt = obj.get("prompt") or obj.get("caption")
                if isinstance(prompt, str) and prompt.strip():
                    return prompt.strip()
            except json.JSONDecodeError:
                continue

        try:
            obj = json.loads(cleaned)
            prompt = obj.get("prompt") or obj.get("caption")
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()
        except json.JSONDecodeError:
            pass

        if cleaned:
            return cleaned

        raise ValueError("LLM response did not contain a valid prompt")

    def generate_prompt(self, media_path: str, source_caption: str = "") -> str:
        """Generate a prompt from configured variables and optional source caption."""
        user_content = self._build_user_content(source_caption=source_caption)
        last_error = None

        for attempt in range(self.retry + 1):
            try:
                self.logger.debug(
                    "Calling caption LLM for %s with variables: %s",
                    media_path,
                    self.variables,
                )
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    stream=self.stream,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )

                result = ""
                if self.stream:
                    reasoning_buffer = ""
                    for chunk in completion:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta.content is not None:
                            result += delta.content
                        reasoning_delta = getattr(delta, "reasoning_content", None)
                        if reasoning_delta is not None:
                            reasoning_buffer += reasoning_delta
                    if not result.strip() and reasoning_buffer.strip():
                        result = reasoning_buffer
                else:
                    if completion.choices and completion.choices[0].message.content:
                        result = completion.choices[0].message.content
                    elif completion.choices:
                        reasoning_text = getattr(
                            completion.choices[0].message, "reasoning_content", None
                        )
                        if reasoning_text and reasoning_text.strip():
                            self.logger.warning(
                                "Empty content; falling back to reasoning_content. "
                                "Endpoint may have ignored enable_thinking=false."
                            )
                            result = reasoning_text

                prompt = self._extract_prompt(result)
                self.logger.info("LLM caption generated successfully")
                return prompt
            except Exception as e:
                last_error = e
                self.logger.warning(
                    "Caption LLM attempt %d/%d failed: %s",
                    attempt + 1,
                    self.retry + 1,
                    e,
                )

        raise RuntimeError(
            f"Caption generation failed after {self.retry + 1} attempts: {last_error}"
        ) from last_error

    def get_caption(self, media_path: str) -> str:
        """Generate a prompt from configured variables. media_path is unused."""
        return self.generate_prompt(media_path=media_path)
