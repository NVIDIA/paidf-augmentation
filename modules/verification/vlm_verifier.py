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

from generation.adapters.openai_chat import OpenAIChatAdapter
from generation.utils import decode_failure_hint
from aug_utils.common import validate_and_cast_config_params
from aug_utils.constants import IMAGE_EXTENSIONS


def _strip_reasoning(content: str) -> str:
    """Remove ``<think>...</think>`` chain-of-thought from a reasoning VLM reply.

    Return only the text after the final ``</think>``. If the block was left
    open (``<think>`` with no close, e.g. truncated by ``max_tokens``), drop it
    entirely. Non-reasoning replies pass through untouched.
    """
    if not content:
        return ""
    cleaned = re.sub(
        r"<think>.*?</think>", " ", content, flags=re.DOTALL | re.IGNORECASE
    )
    lower = cleaned.lower()
    if "</think>" in lower:
        cleaned = cleaned[lower.rfind("</think>") + len("</think>") :]
    elif "<think>" in lower:  # unterminated (truncated) block -> discard it
        cleaned = cleaned[: lower.rfind("<think>")]
    return cleaned.strip()


def _extract_mcq_letter(content: str) -> str:
    """Extract the chosen MCQ option letter (A-D) from a VLM reply.

    Robust to *reasoning* VLMs (e.g. the Qwen3.x "thinking" models) whose reply
    is a chain-of-thought that enumerates the options ("A) ... B) ...") before
    stating the verdict. A naive "first A-D letter" scan returns the "A)" of the
    option list every single time -- which silently pins every answer to A.
    Instead we (1) drop any ``<think>...</think>`` block, then (2) take the
    *last* standalone A-D letter, which is where instruct and reasoning models
    alike place their final answer. Returns ``"UNKNOWN"`` if none is found.
    """
    if not content:
        return "UNKNOWN"
    # Scan ONLY the reasoning-stripped text. A raw fallback would re-read letters
    # from inside a <think> block -- a reasoning-only or truncated reply such as
    # "<think>A wrong, B maybe, C no, D uncertain</think>" would be mistaken for
    # answer "D" (a false pass). No standalone A-D letter after stripping means
    # the model never committed to an answer -> UNKNOWN.
    matches = re.findall(r"\b([A-D])\b", _strip_reasoning(content).upper())
    return matches[-1] if matches else "UNKNOWN"


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
        api_key_env: Optional[str] = None,
        frames: int = 1,
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
        # Number of frames to sample from a video (1 = first frame only, the
        # legacy behavior). >1 samples evenly across the clip so a mid-video
        # event (e.g. a fall) is actually visible to the VLM.
        self.frames = max(1, int(frames))

        # Route chat.completions through the shared one-client adapter. Preserves
        # the prior per-request timeout of 7200s; key resolved from the endpoint's
        # api_key_env env var when supplied, else the role default (VLM_API_KEY).
        self.adapter = OpenAIChatAdapter.for_chat(
            self.endpoint,
            self.model,
            self.logger,
            role="vlm",
            timeout=7200,
            api_key_env=api_key_env,
        )

    @classmethod
    def from_config(
        cls,
        config_params: dict,
        endpoint: str,
        model: str,
        logger: logging.Logger,
        api_key_env: Optional[str] = None,
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
            api_key_env=api_key_env,
            frames=int(config_params.get("frames", 1) or 1),
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
                    # Report the write result: this is the fallback path, so
                    # claiming success on a failed write hands the caller a
                    # path it will open and fail on.
                    if not cv2.imwrite(output_image_path, bgr):
                        self.logger.error(
                            f"Failed to write first frame image: {output_image_path}"
                        )
                        return False
                    self.logger.debug(f"Extracted first frame to: {output_image_path}")
                    return True
                self.logger.error(f"Failed to read first frame from: {video_path}")
                return False

        except Exception as e:
            self.logger.error(f"Error extracting first frame: {e}")
            hint = decode_failure_hint(str(e), video_path)
            if hint:
                self.logger.error(hint)
            return False

    def _extract_frames(
        self, video_path: str, n_frames: int, out_dir: str
    ) -> list[str]:
        """Extract ``n_frames`` evenly-spaced frames from a video.

        Returns the list of saved JPEG paths in chronological order. ``n_frames``
        == 1 yields just the first frame (legacy behavior). Frames are spaced
        across the full clip so a mid-video event is captured.

        Decoding goes through PyAV, not ``cv2.VideoCapture``: OpenCV is built
        ``WITH_FFMPEG=OFF`` in this image, so VideoCapture cannot open any video
        here and would silently yield zero frames.

        Returns an empty list on failure — the caller falls back to the
        first-frame extractor — but logs the underlying cause, which is
        otherwise lost when that fallback fails for the same reason.
        """
        paths: list[str] = []
        try:
            with av.open(video_path) as container:
                if not container.streams.video:
                    self.logger.error(f"No video stream in: {video_path}")
                    return paths
                stream = container.streams.video[0]
                total = self._frame_count(video_path, stream)

                if n_frames <= 1:
                    wanted = {0}
                elif total > 0:
                    wanted = {
                        round(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)
                    }
                else:
                    # Length still unknown: take the first n_frames rather than
                    # nothing, but say so — these are not evenly spaced, so a
                    # mid-clip event can be missed.
                    self.logger.warning(
                        f"Unknown frame count for {video_path}; sampling the "
                        f"first {n_frames} frames instead of evenly-spaced ones"
                    )
                    wanted = None

                for idx, frame in enumerate(container.decode(stream)):
                    if wanted is not None and idx not in wanted:
                        continue
                    bgr = frame.to_ndarray(format="bgr24")
                    p = os.path.join(out_dir, f"frame_{len(paths):02d}.jpg")
                    # Only record frames that actually landed on disk: the caller
                    # opens every returned path, so a silently-failed write would
                    # surface later as FileNotFoundError instead of falling back.
                    if cv2.imwrite(p, bgr):
                        paths.append(p)
                    else:
                        self.logger.error(f"Failed to write frame image: {p}")
                    if len(paths) >= n_frames:
                        break
        except Exception as e:
            self.logger.error(
                f"Error extracting frames from {video_path}: {type(e).__name__}: {e}"
            )
            hint = decode_failure_hint(str(e), video_path)
            if hint:
                self.logger.error(hint)
        return paths

    @staticmethod
    def _frame_count(video_path: str, stream) -> int:
        """Exact frame count, or 0 when it cannot be determined.

        ``stream.frames`` is 0 for containers that do not store ``nb_frames``.
        The sample indices are then used as exact positions, so the fallback has
        to be exact too:

        - ``duration x average_rate`` is an estimate; an off-by-one drops the
          final frame (where a generated event usually resolves) or silently
          decodes the whole clip without filling the request.
        - Packet count is not a safe proxy either: one packet can carry several
          frames (VP9 superframes, DivX packed B-frames, Matroska lacing).

        So count decoded frames. That costs one extra pass, but only for
        containers missing ``nb_frames``, and frames are not converted here so
        it stays cheaper than the sampling pass that follows.
        """
        if stream.frames:
            return int(stream.frames)
        try:
            with av.open(video_path) as probe:
                return sum(1 for _ in probe.decode(probe.streams.video[0]))
        except Exception:
            return 0

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
                        image_paths = [temp_image_path]
                    else:
                        # Download video and sample frames across the clip.
                        temp_video_path = os.path.join(temp_dir, "video.mp4")
                        with (
                            msc.open(media_path, "rb") as src,
                            open(temp_video_path, "wb") as dst,
                        ):
                            shutil.copyfileobj(src, dst)
                        self.logger.debug(f"Downloaded video to {temp_video_path}")

                        image_paths = self._extract_frames(
                            temp_video_path, self.frames, temp_dir
                        )
                        if not image_paths:
                            # Fall back to the av-based first-frame extractor.
                            fallback = os.path.join(temp_dir, "first_frame.jpg")
                            if self._extract_first_frame(temp_video_path, fallback):
                                image_paths = [fallback]
                            else:
                                raise Exception(
                                    "Failed to extract any frame from video"
                                )

                    # Encode each frame as base64.
                    image_items = []
                    for p in image_paths:
                        with open(p, "rb") as image_file:
                            b64 = base64.b64encode(image_file.read()).decode()
                        image_items.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            }
                        )

                    # Format the question
                    formatted_question = (
                        self._format_question_with_reasoning(question, options)
                        if request_reasoning
                        else self._format_question(question, options)
                    )
                    # When multiple frames are sent, tell the VLM they are an
                    # ordered sequence from one clip (so it reasons across time).
                    text = formatted_question
                    if len(image_items) > 1:
                        text = (
                            f"You are shown {len(image_items)} frames sampled in "
                            f"chronological order from a single generated video; "
                            f"consider all of them when answering.\n\n"
                            f"{formatted_question}"
                        )

                    conversation = [
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": text}, *image_items],
                        },
                    ]

                    self.logger.debug("Sending chat completion request with image")
                    use_max_tokens = (
                        max(self.max_tokens, 256)
                        if request_reasoning
                        else self.max_tokens
                    )
                    chat_response = self.adapter.chat(
                        model=self.model,
                        messages=conversation,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        max_tokens=use_max_tokens,
                        stream=self.stream,
                        # Reasoning VLMs (Qwen3.x) otherwise spend the whole (small)
                        # MCQ token budget on chain-of-thought and never reach the
                        # answer letter -- the unterminated <think> block is dropped
                        # and parsing yields UNKNOWN. Disable thinking for the MCQ
                        # answer, mirroring generate_description() below. Harmless for
                        # models without the flag; the think-aware parser still strips
                        # any residual block as a safety net.
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    assistant_message = chat_response.choices[0].message
                    self.logger.debug(f"VLM response: {assistant_message.content}")

                    content = (assistant_message.content or "").strip()
                    vlm_reasoning: Optional[str] = None

                    # Answer extraction is shared (think-aware, last-letter) so it
                    # works for both instruct and reasoning VLMs. See
                    # :func:`_extract_mcq_letter`.
                    vlm_answer = _extract_mcq_letter(content)

                    if request_reasoning and content:
                        # Surface the model's explanation (text up to its final
                        # A-D letter) for callers that want it (e.g. multi-view
                        # consistency checks). Strip any <think> block first so the
                        # surfaced reasoning never leaks the chain-of-thought we
                        # already exclude from answer parsing.
                        reasoning_text = _strip_reasoning(content) or content
                        answer_matches = list(
                            re.finditer(r"\b([A-D])\b", reasoning_text.upper())
                        )
                        if answer_matches:
                            vlm_reasoning = (
                                reasoning_text[: answer_matches[-1].start()].strip()
                                or None
                            )
                        else:
                            vlm_reasoning = reasoning_text
                    elif vlm_answer == "UNKNOWN":
                        self.logger.warning(
                            f"Could not parse clear answer from VLM response: {content}"
                        )

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
        max_tokens_description: int = 512,
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
                chat_response = self.adapter.chat(
                    model=self.model,
                    messages=conversation,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    max_tokens=max_tokens_description,
                    stream=False,
                    # A caption must be prose, not chain-of-thought. Reasoning
                    # VLMs (Qwen3.x) would otherwise return their scratch-work
                    # (and can burn the whole token budget on it, truncating the
                    # actual sentence). Ask the chat template to skip thinking;
                    # harmless for models that don't define the flag. We also
                    # strip any residual <think> block below as a safety net.
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = (chat_response.choices[0].message.content or "").strip()
                content = _strip_reasoning(raw) or raw
                self.logger.debug(f"VLM description: {content[:100]}...")
                return content
            except Exception as e:
                self.logger.error(f"Failed to generate description: {e}", exc_info=True)
                return ""
