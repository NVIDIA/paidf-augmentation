# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import shutil
import subprocess
import tempfile

import multistorageclient as msc

from aug_utils.common import is_remote_path
from cosmos_oss.init import init_environment
from pathlib import Path
from typing import Dict, Optional, Tuple

from .torchrun_utils import reserve_free_port


class LocalCosmosPredictExecutor:
    """
    Local executor for Cosmos Predict 2.5 models.

    Runs inference as a torchrun subprocess, supporting text2world,
    image2world, and video2world inference types.
    """

    def __init__(
        self,
        inference_type: str,
        seed: int,
        guidance: float,
        num_steps: int,
        num_output_frames: Optional[int],
        enable_autoregressive: Optional[bool],
        chunk_size: Optional[int],
        chunk_overlap: Optional[int],
        resolution: str,
        logger: logging.Logger,
        inference_name: str,
        num_processes: int = 8,
        master_port: Optional[int] = 12341,
        torchrun_bin: str = "torchrun",
        inference_script: str = "/opt/cosmos-predict2.5/examples/inference.py",
        model: Optional[str] = None,
        torchrun_timeout: int = 3600,
        offload_tokenizer: Optional[bool] = None,
        offload_text_encoder: Optional[bool] = None,
    ):
        if inference_type not in ("text2world", "image2world", "video2world"):
            raise ValueError(
                f"Unsupported inference_type: {inference_type}. "
                f"Valid types: text2world, image2world, video2world"
            )
        self.inference_type = inference_type
        self.seed = seed
        self.guidance = guidance
        self.num_steps = num_steps
        self.num_output_frames = num_output_frames
        self.enable_autoregressive = enable_autoregressive
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.resolution = resolution
        self.logger = logger
        self.inference_name = inference_name
        self.num_processes = max(1, int(num_processes))
        if master_port is not None:
            master_port = int(master_port)
            if not (1 <= master_port <= 65535):
                raise ValueError(f"invalid master_port: {master_port}, must be 1-65535")
        self.master_port = master_port
        self.torchrun_bin = torchrun_bin
        self.inference_script = inference_script
        self.model = model
        self.torchrun_timeout = int(torchrun_timeout)
        self.cosmos_root = str(Path(self.inference_script).resolve().parent.parent)
        self.offload_tokenizer = bool(offload_tokenizer) if offload_tokenizer else False
        self.offload_text_encoder = bool(offload_text_encoder) if offload_text_encoder else False

        init_environment()

    def _build_predict_params(
        self, prompt_path: str, input_path: Optional[str] = None
    ) -> dict:
        """Build Cosmos Predict 2.5 inference spec JSON."""
        params: Dict = {
            "inference_type": self.inference_type,
            "name": self.inference_name,
            "prompt_path": str(Path(prompt_path).resolve()),
            "num_steps": self.num_steps,
            "seed": self.seed,
            "guidance": self.guidance,
        }

        if input_path is not None:
            params["input_path"] = str(Path(input_path).resolve())

        if self.resolution is not None:
            params["resolution"] = self.resolution
        if self.num_output_frames is not None:
            params["num_output_frames"] = self.num_output_frames
        if self.enable_autoregressive is not None:
            params["enable_autoregressive"] = self.enable_autoregressive
        if self.chunk_size is not None:
            params["chunk_size"] = self.chunk_size
        if self.chunk_overlap is not None:
            params["chunk_overlap"] = self.chunk_overlap

        return params

    # Matches upstream cosmos_predict2/config.py INPUT_EXTENSIONS:
    # both image2world and video2world accept image + video extensions.
    _SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".mp4"}

    def _validate_input_type(self, input_media_path: Optional[str]) -> None:
        """Validate that inference_type matches the input media file type."""
        if self.inference_type == "text2world":
            if input_media_path is not None:
                self.logger.warning(
                    f"inference_type='text2world' ignores input media, "
                    f"but input was provided: {input_media_path}"
                )
            return

        if input_media_path is None:
            raise ValueError(
                f"inference_type='{self.inference_type}' requires an input file, "
                f"but no input media was provided"
            )

        ext = Path(input_media_path).suffix.lower()
        if ext not in self._SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"inference_type='{self.inference_type}' got unsupported file "
                f"extension '{ext}'. Supported: "
                f"{', '.join(sorted(self._SUPPORTED_EXTENSIONS))}"
            )

    def _run_torchrun_inference(self, spec_path: Path, output_dir: Path) -> None:
        """Run Cosmos Predict inference via torchrun subprocess."""
        if not Path(self.inference_script).exists():
            raise FileNotFoundError(
                f"Inference script not found: {self.inference_script}"
            )

        if self.master_port is not None:
            port = self.master_port
            reserved_socket = None
        else:
            port, reserved_socket = reserve_free_port()

        command = [
            self.torchrun_bin,
            "--nproc_per_node",
            str(self.num_processes),
            "--master_port",
            str(port),
            self.inference_script,
            "-i",
            str(spec_path),
            "-o",
            str(output_dir),
        ]

        if self.model:
            command.extend(["--model", self.model])
        if self.offload_tokenizer:
            command.append("--offload_tokenizer")
        if self.offload_text_encoder:
            command.append("--offload_text_encoder")

        self.logger.info(
            f"Running local Cosmos Predict via subprocess: {' '.join(command)}"
        )
        if reserved_socket is not None:
            reserved_socket.close()

        try:
            subprocess.run(
                command,
                check=True,
                env=os.environ.copy(),
                cwd=self.cosmos_root,
                timeout=self.torchrun_timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"torchrun inference timed out after {self.torchrun_timeout} s "
                f"(cwd={self.cosmos_root}, cmd={' '.join(command)})"
            ) from e

    def _copy_file_msc(self, src_path: str, dst_path: str) -> bool:
        """Copy a file using multistorageclient."""
        try:
            with (
                msc.open(src_path, "rb") as src,
                msc.open(dst_path, "wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
            self.logger.debug(f"Copied {src_path} to {dst_path}")
            return True
        except OSError as e:
            self.logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
            return False

    def _download_to_local(
        self, remote_path: str, local_dir: Path, filename: str
    ) -> str:
        """Download a remote file to a local directory."""
        local_path = local_dir / filename
        self.logger.info(f"Downloading {remote_path} to {local_path}")
        with (
            msc.open(remote_path, "rb") as src,
            open(local_path, "wb") as dst,
        ):
            shutil.copyfileobj(src, dst)
        self.logger.debug(f"Downloaded {remote_path} to {local_path}")
        return str(local_path)

    def execute(
        self,
        prompt: str,
        input_media_path: Optional[str],
        output_path: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute Cosmos Predict video generation.

        Args:
            prompt: Text prompt for generation
            input_media_path: Path to input video/image (None for text2world)
            output_path: Path to save the generated video

        Returns:
            Tuple of (success: bool, output_path: str or None)
        """
        # Validate inference_type matches input media
        self._validate_input_type(input_media_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Download remote input to local temp directory if needed
            local_input_path = input_media_path
            if input_media_path is not None and is_remote_path(input_media_path):
                ext = Path(input_media_path).suffix or ".mp4"
                local_input_path = self._download_to_local(
                    input_media_path, temp_dir_path, f"input_media{ext}"
                )

            # Write prompt to temp file
            prompt_file = temp_dir_path / "prompt.txt"
            with open(prompt_file, "w") as f:
                f.write(prompt)

            # Build inference spec
            local_params = self._build_predict_params(
                str(prompt_file), local_input_path
            )
            self.logger.info(f"Predict params: {local_params}")

            try:
                spec_path = temp_dir_path / "inference_spec.json"
                with open(spec_path, "w") as f:
                    json.dump(local_params, f, indent=2)
                self._run_torchrun_inference(spec_path, temp_dir_path)
            except subprocess.CalledProcessError as e:
                self.logger.error(
                    f"torchrun subprocess failed with return code {e.returncode}"
                )
                return False, None
            except RuntimeError as e:
                self.logger.error(f"Inference failed: {e}")
                return False, None
            except OSError as e:
                self.logger.error(f"Inference failed: {e}")
                return False, None

            generated_video = temp_dir_path / f"{self.inference_name}.mp4"

            if not generated_video.exists():
                self.logger.error(f"Generated video not found at {generated_video}")
                return False, None

            if not is_remote_path(output_path):
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            if not self._copy_file_msc(str(generated_video), output_path):
                self.logger.error(f"Failed to copy video to {output_path}")
                return False, None
            self.logger.info(f"Copied generated video to {output_path}")

            self.logger.debug(f"Cleaned up temp directory: {temp_dir}")
            return True, output_path
