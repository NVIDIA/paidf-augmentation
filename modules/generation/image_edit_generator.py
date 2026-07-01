# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
from typing import Dict, Tuple, Optional

import multistorageclient as msc
from openai import OpenAI

from .base import BaseGenerator
from aug_utils.common import validate_and_cast_config_params

# Fallback per-request timeout (seconds) when none is supplied (the configured
# value comes from pipeline.request_timeout). A wedged/slow endpoint fails after
# this instead of hanging; pipeline.retry then decides whether to re-call.
DEFAULT_REQUEST_TIMEOUT = 120.0


class ImageEditGenerator(BaseGenerator):
    """Image edit model generator for image-to-image editing via an OpenAI-compatible endpoint."""

    _PARAM_TYPES = {
        "num_inference_steps": int,
        "guidance_scale": float,
        "seed": int,
        "endpoint": str,
        "model": str,
    }

    def __init__(
        self,
        endpoint: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        logger: logging.Logger,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        negative_prompt: Optional[str] = " ",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(logger)

        params = {
            "endpoint": endpoint,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        if model is not None:
            params["model"] = model

        validated_params = validate_and_cast_config_params(
            params, self._PARAM_TYPES, logger
        )

        self.endpoint = validated_params["endpoint"].rstrip("/")
        self.num_inference_steps = validated_params["num_inference_steps"]
        self.guidance_scale = validated_params["guidance_scale"]
        self.seed = validated_params["seed"]
        self.model = validated_params.get("model")
        self.negative_prompt = negative_prompt
        self.timeout = DEFAULT_REQUEST_TIMEOUT if timeout is None else float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0 seconds")

        # Some endpoints don't require auth; "dummy" is a placeholder that
        # makes the OpenAI client happy when no key is configured.
        # max_retries=0: a wedged/slow endpoint should fail after `timeout`
        # rather than the SDK silently re-issuing the request; pipeline.retry
        # is the single source of retries.
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=api_key if api_key else "dummy",
            timeout=self.timeout,
            max_retries=0,
        )

        self.logger.info(
            f"Image edit generator initialized with endpoint: {self.endpoint} "
            f"(request timeout {self.timeout:.0f}s, no client-side retries)"
        )

    def get_required_inputs(self) -> Dict[str, bool]:
        """Image edit only requires the RGB input image."""
        return {
            "rgb": True,
        }

    def execute(
        self,
        prompt: str,
        input_media_path: str,
        control_inputs: Dict[str, str],
        output_path: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute image edit generation.

        Args:
            prompt: Text instruction for image editing (e.g., "Convert to watercolor")
            input_media_path: Path to input image
            control_inputs: Ignored (no control inputs needed)
            output_path: Path to save generated image

        Returns:
            Tuple of (success: bool, output_path: str or None)
        """
        response = None
        request_id = None
        try:
            self.logger.debug(f"Reading input image: {input_media_path}")
            with msc.open(input_media_path, "rb") as f:
                image_bytes = f.read()

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            ext = os.path.splitext(input_media_path)[1].lower()
            mime_type = "image/png"
            if ext in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            elif ext == ".webp":
                mime_type = "image/webp"

            self.logger.info(
                f"Image edit request starting -> output_path={output_path}"
            )
            self.logger.debug(f"Calling image edit API with prompt: {prompt}")
            extra_body_payload = {
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "seed": self.seed,
            }
            if self.negative_prompt is not None:
                extra_body_payload["negative_prompt"] = self.negative_prompt

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                extra_body={"extra_body": extra_body_payload},
            )
            request_id = getattr(response, "id", None)
            self.logger.info(
                f"Image edit response received request_id={request_id} -> output_path={output_path}"
            )
            self.logger.debug(f"Response received, type: {type(response)}")
            self.logger.debug(
                f"Response structure: choices[0].message.content type: {type(response.choices[0].message.content)}"
            )
            if response.choices[0].message.content:
                self.logger.debug(
                    f"Content[0] type: {type(response.choices[0].message.content[0])}"
                )

            # Response content can be either object attributes or dict
            try:
                image_url = response.choices[0].message.content[0].image_url.url
            except AttributeError as err:
                content_item = response.choices[0].message.content[0]
                if isinstance(content_item, dict):
                    image_url = content_item.get("image_url", {}).get("url")
                else:
                    raise TypeError(
                        f"Unexpected content structure: {type(content_item)}"
                    ) from err

            if not image_url:
                self.logger.error(
                    f"No image URL in response -> output_path={output_path} request_id={request_id}"
                )
                return False, None

            # Parse data URL: "data:image/png;base64,<base64_data>"
            if "base64," in image_url:
                output_b64 = image_url.split("base64,")[1]
            else:
                self.logger.error(
                    f"Unexpected response format (no base64) -> output_path={output_path} request_id={request_id}"
                )
                return False, None

            output_bytes = base64.b64decode(output_b64)

            with msc.open(output_path, "wb") as f:
                f.write(output_bytes)

            self.logger.info(f"Successfully generated image: {output_path}")
            return True, output_path

        except Exception as e:
            rid = getattr(response, "id", None) if response is not None else None
            self.logger.error(
                f"Image edit failed -> output_path={output_path} request_id={rid} error={e}",
                exc_info=True,
            )
            return False, None
