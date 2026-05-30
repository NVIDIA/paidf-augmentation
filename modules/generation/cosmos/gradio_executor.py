# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
from typing import List, Dict, Any, Optional
from gradio_client import Client, handle_file
import tempfile
import logging
import shutil
import requests
from urllib.parse import urlparse

import multistorageclient as msc

from aug_utils.common import capture_prints


class GradioCosmosExecutor:
    """
    Gradio client for Cosmos Transfer models (CT2.5).

    Supports two model versions with different parameter formats:
        - CT2.5: name, prompt_path (file), video_path, guidance, num_steps, seed, sigma_max (str), modalities
    """

    def __init__(
        self,
        endpoint: str,
        sigma: int,
        seed: int,
        guidance: int,
        num_steps: int,
        modalities: List[str],
        weights: Dict[str, float],
        positive_prompt: str,
        negative_prompt: str,
        logger: logging.Logger,
        inference_name: str,
        model_version: Optional[str] = None,  # Default to "ct2.5" if None
    ):
        self.endpoint = endpoint
        self.sigma = sigma
        self.seed = seed
        self.guidance = guidance
        self.num_steps = num_steps
        self.modalities = modalities
        self.weights = weights
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.logger = logger
        self.inference_name = inference_name

        # Default to ct2.5, or use explicitly configured version
        if model_version is None:
            self.model_version = "ct2.5"
        else:
            # Convert "ct25" to "ct2.5" for internal use
            self.model_version = (
                "ct2.5" if model_version.lower() == "ct25" else model_version.lower()
            )

        self.logger.info(f"Using Cosmos Transfer version: {self.model_version}")

        # Configure Gradio client with auth and extended timeouts for video generation
        client_kwargs = {
            "verbose": True,
            "httpx_kwargs": {
                "timeout": 1800.0,  # 30 minutes for long video generation
            },
        }

        # Use VLM_API_KEY or LLM_API_KEY for Gradio auth when available
        gradio_api_key = os.getenv("VLM_API_KEY") or os.getenv("LLM_API_KEY")
        if gradio_api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {gradio_api_key}"}
            self.logger.info("Using API key for Gradio auth")

        self.client = Client(self.endpoint, **client_kwargs)

    def _upload_to_gradio(self, content: bytes, filename: str, temp_dir: str) -> str:
        """Upload content to Gradio server and return the uploaded path."""
        local_path = os.path.join(temp_dir, filename)
        with open(local_path, "wb" if isinstance(content, bytes) else "w") as f:
            f.write(content)
        return json.loads(
            self.client.predict(handle_file(local_path), api_name="/upload_file")
        )["path"]

    def _build_ct25_params(
        self,
        prompt_path: str,
        video: str,
        controls: Dict[str, Optional[str]],
        available: Dict[str, Optional[str]],
    ) -> dict:
        """Build CT2.5 format: name, prompt_path, video_path, modalities with control paths."""
        params = {
            "name": self.inference_name,
            "prompt_path": prompt_path,
            "video_path": video,
            "guidance": self.guidance,
            "num_steps": self.num_steps,
            "seed": self.seed,
        }

        # Optional sigma_max as string
        if self.sigma and self.sigma > 0:
            params["sigma_max"] = str(self.sigma)

        # Add modalities with control_path
        for mod in ["edge", "depth", "seg", "vis"]:
            if mod in available:
                control_path = self._normalize_control_path(controls.get(mod))
                params[mod] = {"control_weight": self.weights.get(mod, 1.0)}
                if control_path is not None:
                    params[mod]["control_path"] = control_path

        return params

    @staticmethod
    def _normalize_control_path(path: Optional[str]) -> Optional[str]:
        """Convert empty/null-like control paths to None."""
        if path is None:
            return None
        if isinstance(path, str) and path.strip().lower() in {"", "none", "null"}:
            return None
        return path

    def _prepare_gradio_params(
        self, prompt: str, input_video_path: str, control_videos: Dict[str, str]
    ) -> dict:
        """Upload files and build parameters for CT2.5."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Upload input video (read from storage)
            if not msc.is_file(input_video_path):
                raise Exception(f"Input video not found: {input_video_path}")
            with msc.open(input_video_path, "rb") as f:
                content = f.read()
            video = self._upload_to_gradio(content, "input.mp4", temp_dir)

            # Upload control videos (read from storage)
            controls = {}
            for mod, path in control_videos.items():
                normalized_path = self._normalize_control_path(path)
                if normalized_path is None:
                    self.logger.debug(
                        f"Control video for {mod} is None, skipping upload"
                    )
                    controls[mod] = None
                    continue

                if not msc.is_file(normalized_path):
                    raise Exception(f"Control video not found: {normalized_path}")
                with msc.open(normalized_path, "rb") as f:
                    content = f.read()
                controls[mod] = self._upload_to_gradio(content, f"{mod}.mp4", temp_dir)

            # Build params
            if self.model_version == "ct2.5":
                prompt_path = self._upload_to_gradio(
                    prompt.encode(), "prompt.txt", temp_dir
                )
                return self._build_ct25_params(
                    prompt_path, video, controls, control_videos
                )
            else:
                raise Exception(f"Unsupported model version: {self.model_version}")

    def _parse_video_path_from_result(self, result: Any) -> Optional[str]:
        """
        Parse the video path from Gradio result's JSON message.

        Args:
            result: Gradio API result tuple (dict, message_string)

        Returns:
            Server video path or None if not found
        """
        try:
            if not isinstance(result, (list, tuple)) or len(result) < 2:
                self.logger.error("Invalid result structure")
                return None

            message = result[1]
            if not isinstance(message, str):
                self.logger.error("Result message is not a string")
                return None

            # Look for the JSON block in the message
            json_start = message.find("Result json: {")
            if json_start < 0:
                self.logger.error("No 'Result json:' found in message")
                return None

            # Parse JSON
            json_str = message[json_start + len("Result json: ") :]
            parsed = json.loads(json_str)

            # Extract video path from 'videos' list
            if "videos" not in parsed:
                self.logger.error("No 'videos' key in JSON result")
                return None

            if not isinstance(parsed["videos"], list) or len(parsed["videos"]) == 0:
                self.logger.error("'videos' is not a valid list")
                return None

            video_path = parsed["videos"][0]
            self.logger.debug(f"Found server output path: {video_path}")

            # Convert relative path to absolute path
            if not video_path.startswith("/"):
                video_path = f"/workspace/{video_path}"
                self.logger.debug(f"Converted to absolute path: {video_path}")

            return video_path

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from message: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing video path: {e}")
            return None

    def _download_video_from_server(
        self, server_video_path: str, local_output_path: str
    ) -> bool:
        """
        Download video file from Gradio server to local path.

        Args:
            server_video_path: Absolute path to video file on server
            local_output_path: Local path where the file should be saved

        Returns:
            True if download successful, False otherwise
        """
        try:
            # Extract base URL from endpoint
            parsed_endpoint = urlparse(self.endpoint)
            base_url = f"{parsed_endpoint.scheme}://{parsed_endpoint.netloc}"

            # Construct the Gradio file serving URL
            file_url = f"{base_url}/gradio_api/file={server_video_path}"
            self.logger.debug(f"Downloading file from: {file_url}")

            # Download the file with timeout for large files
            response = requests.get(file_url, timeout=300)
            response.raise_for_status()

            # Save to local path
            with open(local_output_path, "wb") as f:
                f.write(response.content)

            file_size_mb = len(response.content) / (1024 * 1024)
            self.logger.debug(f"Successfully downloaded file ({file_size_mb:.2f} MB)")
            return True

        except requests.RequestException as e:
            self.logger.error(f"HTTP request failed: {e}")
            return self._fallback_direct_copy(server_video_path, local_output_path)

        except Exception as e:
            self.logger.error(f"Unexpected error during download: {e}")
            return False

    def _fallback_direct_copy(self, server_path: str, local_path: str) -> bool:
        """
        Fallback method to copy file directly if server and client share filesystem.

        Args:
            server_path: Path to file on server
            local_path: Local destination path

        Returns:
            True if copy successful, False otherwise
        """
        try:
            if os.path.exists(server_path):
                self.logger.info(f"Fallback: Copying file directly from {server_path}")
                shutil.copy2(server_path, local_path)
                return True
            else:
                self.logger.error(f"Cannot access file at {server_path}")
                return False
        except Exception as e:
            self.logger.error(f"Direct copy failed: {e}")
            return False

    def _capture_gradio_output(self):
        """
        Context manager to capture Gradio client output and redirect to logger.
        """
        return capture_prints(self.logger)

    def execute(
        self,
        prompt: str,
        input_video_path: str,
        control_videos: Dict[str, str],
        output_path: str,
    ):
        """
        Execute Cosmos Transfer video generation.

        Args:
            prompt: Text prompt for generation
            input_video_path: Path to input video
            control_videos: Dictionary of control modality videos
            output_path: Path to save the generated video

        Returns:
            Tuple of (success: bool, output_path: str or None)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare and send request to Gradio server
            gradio_params = self._prepare_gradio_params(
                prompt, input_video_path, control_videos
            )
            self.logger.debug(f"Gradio params: {gradio_params}")

            with self._capture_gradio_output():
                # Call Gradio API
                result = self.client.predict(
                    request_text=json.dumps(gradio_params), api_name="/generate_video"
                )

                # Parse video path from result
                server_video_path = self._parse_video_path_from_result(result)
                if not server_video_path:
                    self.logger.error("Failed to parse video path from result")
                    return False, None

                # Download video from server
                local_output_path = os.path.join(temp_dir, "output.mp4")
                if not self._download_video_from_server(
                    server_video_path, local_output_path
                ):
                    self.logger.error("Failed to download video from server")
                    return False, None

                # Write video to final output location (via storage)
                try:
                    with (
                        open(local_output_path, "rb") as src,
                        msc.open(output_path, "wb") as dst,
                    ):
                        shutil.copyfileobj(src, dst)

                    self.logger.debug(f"Successfully saved video to {output_path}")
                    return True, output_path

                except Exception as e:
                    self.logger.error(f"Failed to write output file: {e}")
                    return False, None
