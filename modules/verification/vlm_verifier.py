# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import tempfile
import os
import re
import base64
import shutil
import av
import cv2
from typing import Optional


import multistorageclient as msc
from openai import OpenAI

from aug_utils.common import validate_and_cast_config_params
from aug_utils.constants import IMAGE_EXTENSIONS


class VLMVerifier:
    """Verify attributes using VLM on images or video first frames."""

    # Define expected parameter types for validation
    _PARAM_TYPES = {
        "retry": int,
        "temperature": float,
        "top_p": float,
        "frequency_penalty": float,
        "max_tokens": int,
        "stream": bool,
        "system_prompt": str,
        "endpoint": str,
        "model": str,
    }

    def __init__(
        self,
        system_prompt: str,
        retry: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        max_tokens: int,
        stream: bool,
        endpoint: str,
        model: str,
        logger: logging.Logger,
    ):
        # Validate and cast parameters to ensure correct types
        params = {
            "system_prompt": system_prompt,
            "retry": retry,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "stream": stream,
            "endpoint": endpoint,
            "model": model,
        }

        validated_params = validate_and_cast_config_params(
            params, self._PARAM_TYPES, logger
        )

        self.system_prompt = validated_params["system_prompt"]
        self.retry = validated_params["retry"]
        self.temperature = validated_params["temperature"]
        self.top_p = validated_params["top_p"]
        self.frequency_penalty = validated_params["frequency_penalty"]
        self.max_tokens = validated_params["max_tokens"]
        self.stream = validated_params["stream"]

        # Ensure endpoint URL is properly formatted
        self.endpoint = validated_params["endpoint"].rstrip("/")
        self.model = validated_params["model"]

        self.logger = logger

    @classmethod
    def from_config(
        cls, config_params: dict, endpoint: str, model: str, logger: logging.Logger
    ):
        """
        Create VLMVerifier instance from configuration dictionary with type validation.

        Args:
            config_params: Dictionary containing VLM configuration parameters
            endpoint: VLM endpoint URL
            model: VLM model name
            logger: Logger instance

        Returns:
            VLMVerifier: Configured instance with validated parameters
        """
        # Extract and validate parameters
        params = {
            "system_prompt": config_params.get("system_prompt", ""),
            "retry": config_params.get("parameters", {}).get("retry", 0),
            "temperature": config_params.get("parameters", {}).get("temperature", 0.0),
            "top_p": config_params.get("parameters", {}).get("top_p", 1.0),
            "frequency_penalty": config_params.get("parameters", {}).get(
                "frequency_penalty", 0.0
            ),
            "max_tokens": config_params.get("parameters", {}).get("max_tokens", 512),
            "stream": config_params.get("parameters", {}).get("stream", False),
            "endpoint": endpoint,
            "model": model,
        }

        validated_params = validate_and_cast_config_params(
            params, cls._PARAM_TYPES, logger
        )

        logger.info(
            f"VLM verifier using endpoint={validated_params['endpoint']}, model={validated_params['model']}"
        )
        return cls(
            system_prompt=validated_params["system_prompt"],
            retry=validated_params["retry"],
            temperature=validated_params["temperature"],
            top_p=validated_params["top_p"],
            frequency_penalty=validated_params["frequency_penalty"],
            max_tokens=validated_params["max_tokens"],
            stream=validated_params["stream"],
            endpoint=validated_params["endpoint"],
            model=validated_params["model"],
            logger=logger,
        )

    def _extract_first_frame(self, video_path: str, output_image_path: str) -> bool:
        """
        Extract the first frame from a video file.

        Args:
            video_path: Path to the video file
            output_image_path: Path to save the extracted frame

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with av.open(video_path) as container:
                if not container.streams.video:
                    self.logger.error(f"No video stream in: {video_path}")
                    return False
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    bgr = frame.to_ndarray(format="bgr24")
                    cv2.imwrite(output_image_path, bgr)
                    self.logger.debug(f"Extracted first frame to: {output_image_path}")
                    return True
                self.logger.error(f"Failed to read first frame from: {video_path}")
                return False

        except Exception as e:
            self.logger.error(f"Error extracting first frame: {e}")
            return False

    def _format_question(self, question: str, options: dict) -> str:
        """
        Format the question with options for the VLM.

        Args:
            question: The question text
            options: Dictionary of option letters to values

        Returns:
            str: Formatted question string
        """
        options_text = "\n".join([f"{k}) {v}" for k, v in sorted(options.items())])
        return f"{question}\n{options_text}\n\nAnswer with only a single letter (A, B, C, or D)."

    def _format_question_with_reasoning(self, question: str, options: dict) -> str:
        """Format the question when reasoning is requested (e.g. for multi-view consistency)."""
        options_text = "\n".join([f"{k}) {v}" for k, v in sorted(options.items())])
        return (
            f"{question}\n{options_text}\n\n"
            "First provide a brief explanation (1-2 sentences) of why the image is consistent or inconsistent across views. "
            "Then on the last line give only your answer as a single letter (A, B, C, or D)."
        )

    def verify_question(
        self,
        media_path: str,
        question: str,
        options: dict,
        correct_answer: str,
        request_reasoning: bool = False,
    ) -> tuple[bool, str, Optional[str]]:
        """
        Verify a single question by asking the VLM about an image or video first frame.

        Args:
            media_path: Path to the image or video file
            question: The question to ask
            options: Dictionary of answer options
            correct_answer: The expected correct answer letter
            request_reasoning: If True, ask VLM for brief reasoning and return it (e.g. for multi-view consistency).

        Returns:
            Tuple of (is_correct: bool, vlm_answer: str, vlm_reasoning: Optional[str])

        Raises:
            Exception: If verification fails
        """
        retries_remaining = self.retry

        while True:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Check if media file exists
                    if not msc.is_file(media_path):
                        raise Exception(f"Media file not found: {media_path}")

                    # Detect media type by file extension
                    file_ext = os.path.splitext(media_path)[1].lower()
                    is_image = file_ext in IMAGE_EXTENSIONS

                    if is_image:
                        # Download image directly
                        temp_image_path = os.path.join(temp_dir, f"image{file_ext}")
                        with (
                            msc.open(media_path, "rb") as src,
                            open(temp_image_path, "wb") as dst,
                        ):
                            shutil.copyfileobj(src, dst)
                        self.logger.debug(f"Downloaded image to {temp_image_path}")
                    else:
                        # Download video and extract first frame
                        temp_video_path = os.path.join(temp_dir, "video.mp4")
                        with (
                            msc.open(media_path, "rb") as src,
                            open(temp_video_path, "wb") as dst,
                        ):
                            shutil.copyfileobj(src, dst)
                        self.logger.debug(f"Downloaded video to {temp_video_path}")

                        # Extract first frame
                        temp_image_path = os.path.join(temp_dir, "first_frame.jpg")
                        if not self._extract_first_frame(
                            temp_video_path, temp_image_path
                        ):
                            raise Exception("Failed to extract first frame from video")

                    api_key = os.environ.get("VLM_API_KEY")
                    if not api_key:
                        self.logger.warning(
                            "No VLM_API_KEY set. Using 'not-used' as placeholder."
                        )
                        api_key = "not-used"

                    client = OpenAI(
                        base_url=self.endpoint, api_key=api_key, timeout=7200
                    )

                    # Encode image as base64
                    with open(temp_image_path, "rb") as image_file:
                        image_b64 = base64.b64encode(image_file.read()).decode()

                    # Format the question
                    formatted_question = (
                        self._format_question_with_reasoning(question, options)
                        if request_reasoning
                        else self._format_question(question, options)
                    )

                    conversation = [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": formatted_question},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    },
                                },
                            ],
                        },
                    ]

                    self.logger.debug("Sending chat completion request with image")
                    use_max_tokens = (
                        max(self.max_tokens, 256)
                        if request_reasoning
                        else self.max_tokens
                    )
                    chat_response = client.chat.completions.create(
                        model=self.model,
                        messages=conversation,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        max_tokens=use_max_tokens,
                        stream=self.stream,
                    )
                    assistant_message = chat_response.choices[0].message
                    self.logger.debug(f"VLM response: {assistant_message.content}")

                    content = (assistant_message.content or "").strip()
                    vlm_reasoning: Optional[str] = None

                    if request_reasoning and content:
                        # Parse: last standalone letter A-D is the answer; rest is reasoning
                        answer_matches = list(
                            re.finditer(r"\b([A-D])\b", content.upper())
                        )
                        if answer_matches:
                            vlm_answer = answer_matches[-1].group(1)
                            reasoning_end = answer_matches[-1].start()
                            vlm_reasoning = content[:reasoning_end].strip() or None
                        else:
                            vlm_answer = "UNKNOWN"
                            vlm_reasoning = content
                    else:
                        # Extract answer - look for single letter A, B, C, or D
                        answer_match = re.search(r"\b([A-D])\b", content.upper())
                        if answer_match:
                            vlm_answer = answer_match.group(1)
                        else:
                            vlm_answer = content[:1].upper() if content else "UNKNOWN"
                            if vlm_answer not in ["A", "B", "C", "D"]:
                                self.logger.warning(
                                    f"Could not parse clear answer from VLM response: {content}"
                                )
                                vlm_answer = "UNKNOWN"

                    is_correct = vlm_answer == correct_answer.upper()
                    self.logger.debug(
                        f"Question verification: expected={correct_answer}, got={vlm_answer}, correct={is_correct}"
                    )

                    return is_correct, vlm_answer, vlm_reasoning

            except Exception as e:
                self.logger.error(f"Failed to verify question: {e}", exc_info=True)
                if retries_remaining > 0:
                    retries_remaining -= 1
                    self.logger.info(
                        f"Retrying question verification ({retries_remaining} retries remaining)"
                    )
                    continue
                raise

    def generate_description(
        self,
        media_path: str,
        user_prompt: str,
        system_prompt_override: Optional[str] = None,
        max_tokens_description: int = 256,
    ) -> str:
        """
        Ask the VLM to generate a natural-language description of the image (e.g. for natural_caption).
        Used after verification passes to get a VLM-generated caption describing the image and attributes.

        Args:
            media_path: Path to the image or video file
            user_prompt: The prompt asking for a description (e.g. include attribute summary)
            system_prompt_override: Optional system prompt; if None, uses a default description prompt
            max_tokens_description: Max tokens for the description response

        Returns:
            The VLM's text response (natural caption), or empty string on failure.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            if not msc.is_file(media_path):
                self.logger.error(f"Media file not found: {media_path}")
                return ""

            file_ext = os.path.splitext(media_path)[1].lower()
            is_image = file_ext in IMAGE_EXTENSIONS

            if is_image:
                temp_image_path = os.path.join(temp_dir, f"image{file_ext}")
                with (
                    msc.open(media_path, "rb") as src,
                    open(temp_image_path, "wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)
            else:
                temp_video_path = os.path.join(temp_dir, "video.mp4")
                with (
                    msc.open(media_path, "rb") as src,
                    open(temp_video_path, "wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)
                temp_image_path = os.path.join(temp_dir, "first_frame.jpg")
                if not self._extract_first_frame(temp_video_path, temp_image_path):
                    self.logger.error("Failed to extract first frame for description")
                    return ""

            try:
                api_key = os.environ.get("VLM_API_KEY")
                if not api_key:
                    self.logger.warning(
                        "No VLM_API_KEY set. Using 'not-used' as placeholder."
                    )
                    api_key = "not-used"
                client = OpenAI(base_url=self.endpoint, api_key=api_key, timeout=7200)
                with open(temp_image_path, "rb") as image_file:
                    image_b64 = base64.b64encode(image_file.read()).decode()

                system = system_prompt_override or (
                    "You are an expert at describing images of people. "
                    "Write a single natural sentence describing the person's appearance and clothing."
                )
                conversation = [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    },
                ]
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=conversation,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    max_tokens=max_tokens_description,
                    stream=False,
                )
                content = (chat_response.choices[0].message.content or "").strip()
                self.logger.debug(f"VLM description: {content[:100]}...")
                return content
            except Exception as e:
                self.logger.error(f"Failed to generate description: {e}", exc_info=True)
                return ""
