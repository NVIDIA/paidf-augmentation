# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import posixpath
import shutil
import subprocess
import tempfile

import multistorageclient as msc

from aug_utils.common import is_remote_path
from cosmos_oss.init import init_environment
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .torchrun_utils import reserve_free_port
from ..utils import (
    UnsupportedVideoFormatError,
    probe_video_format,
    video_format_unsupported,
)


class LocalCosmosExecutor:
    """
    Local executor for Cosmos Transfer2.5 models.

    Runs inference as a torchrun subprocess.
    """

    def __init__(
        self,
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
        model_version: Optional[str] = None,
        num_processes: int = 1,
        master_port: Optional[int] = 12341,
        torchrun_bin: str = "torchrun",
        inference_script: str = "/opt/cosmos-transfer2.5/examples/inference.py",
        model: Optional[str] = None,
        seg_control_prompt: str = "",
        torchrun_timeout: int = 3600,
    ):
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
        self.model_version = (
            "ct2.5" if model_version in (None, "ct25") else str(model_version).lower()
        )
        self.seg_control_prompt = seg_control_prompt
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

        init_environment()

        if self.model_version != "ct2.5":
            raise ValueError(
                f"Unsupported local Cosmos model version: {self.model_version}"
            )

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
        if "seg" in available and self.seg_control_prompt:
            params["seg"]["control_prompt"] = self.seg_control_prompt

        return params

    @staticmethod
    def _normalize_control_path(path: Optional[str]) -> Optional[str]:
        """Convert empty/null-like control paths to None."""
        if path is None:
            return None
        if isinstance(path, str) and path.strip().lower() in {"", "none", "null"}:
            return None
        return path

    def _prepare_local_params(
        self, prompt_path: str, input_video_path: str, control_videos: Dict[str, str]
    ) -> dict:
        """Build parameters for CT2.5."""
        return self._build_ct25_params(
            prompt_path, input_video_path, control_videos, control_videos
        )

    def _run_torchrun_inference(self, spec_path: Path, output_dir: Path) -> None:
        """Run Cosmos inference via torchrun subprocess."""
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

        self.logger.info(f"Running local Cosmos via subprocess: {' '.join(command)}")
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

    def _fallback_direct_copy(self, server_path: str, local_path: str) -> bool:
        """
        Fallback method to copy file using multistorageclient.

        Args:
            server_path: Path to file on server
            local_path: Local destination path

        Returns:
            True if copy successful, False otherwise
        """
        try:
            if msc.is_file(server_path):
                self.logger.info(f"Fallback: Copying file directly from {server_path}")
                with (
                    msc.open(server_path, "rb") as src,
                    msc.open(local_path, "wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)
                return True
            else:
                self.logger.error(f"Cannot access file at {server_path}")
                return False
        except OSError as e:
            self.logger.error(f"Direct copy failed: {e}")
            return False

    def _copy_file_msc(self, src_path: str, dst_path: str) -> bool:
        """
        Copy a file using multistorageclient.

        Args:
            src_path: Source file path
            dst_path: Destination file path

        Returns:
            True if copy successful, False otherwise
        """
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
        """
        Download a remote file to a local directory.

        Args:
            remote_path: Remote file path (URL or cloud storage)
            local_dir: Local directory to save the file
            filename: Name for the local file

        Returns:
            Local file path
        """
        local_path = local_dir / filename
        self.logger.info(f"Downloading {remote_path} to {local_path}")
        with (
            msc.open(remote_path, "rb") as src,
            open(local_path, "wb") as dst,
        ):
            shutil.copyfileobj(src, dst)
        self.logger.debug(f"Downloaded {remote_path} to {local_path}")
        return str(local_path)

    def _check_input_decodable(self, local_video_path: str) -> None:
        """Raise UnsupportedVideoFormatError if the input can't be decoded here.

        Probes the already-localized path with ffprobe; an indeterminate probe
        does not block the run. See ``generation.utils`` for the rationale.
        """
        fmt = probe_video_format(local_video_path)
        if not fmt:
            self.logger.debug(f"Could not probe {local_video_path}; skipping check.")
            return

        self.logger.info(
            f"Input video format: codec={fmt.get('codec_name')!r} "
            f"profile={fmt.get('profile')!r} pix_fmt={fmt.get('pix_fmt')!r}"
        )
        reason = video_format_unsupported(
            fmt.get("codec_name"), fmt.get("profile"), fmt.get("pix_fmt")
        )
        if reason is not None:
            raise UnsupportedVideoFormatError(
                f"Cannot decode input video {local_video_path!r}: {reason}. This "
                "image decodes H.264 via NVDEC (h264_cuvid) only -- re-encode to "
                "H.264 8-bit 4:2:0 (-pix_fmt yuv420p) before running."
            )

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
            temp_dir_path = Path(temp_dir)

            # Download remote input video to local temp directory if needed
            local_input_video_path = input_video_path
            if is_remote_path(input_video_path):
                local_input_video_path = self._download_to_local(
                    input_video_path, temp_dir_path, "input_video.mp4"
                )

            # Fail fast on inputs the container's NVDEC-only ffmpeg cannot decode
            # (H.264 4:4:4 / 4:2:2 / 10-bit) with a clear, actionable message,
            # instead of letting the torchrun subprocess hang/crash deep inside
            # Cosmos' video reader with a cryptic cuvid decode error.
            try:
                self._check_input_decodable(local_input_video_path)
            except UnsupportedVideoFormatError as e:
                self.logger.error(str(e))
                return False, None

            # Download remote control videos to local temp directory if needed
            local_control_videos = {}
            for mod, control_path in control_videos.items():
                if control_path and is_remote_path(control_path):
                    local_control_videos[mod] = self._download_to_local(
                        control_path, temp_dir_path, f"control_{mod}.mp4"
                    )
                else:
                    local_control_videos[mod] = control_path

            # Prepare and serialize local_params to JSON file in temp_dir
            prompt_file = temp_dir_path / "prompt.txt"
            with open(prompt_file, "w") as f:
                f.write(prompt)

            local_params = self._prepare_local_params(
                str(prompt_file), local_input_video_path, local_control_videos
            )
            self.logger.info(f"Local params: {local_params}")

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
            generated_txt = temp_dir_path / f"{self.inference_name}.txt"

            if not generated_video.exists():
                self.logger.error(f"Generated video not found at {generated_video}")
                return False, None

            if is_remote_path(output_path):
                parsed = urlparse(output_path)
                url_path = parsed.path or output_path
                output_dir = parsed._replace(path=posixpath.dirname(url_path)).geturl()
                output_base = posixpath.splitext(posixpath.basename(url_path))[0]
                if not output_base:
                    raise ValueError(
                        f"Cannot derive output filename from remote path: {output_path!r}"
                    )
            else:
                output_path_obj = Path(output_path)
                output_dir = str(output_path_obj.parent)
                output_base = output_path_obj.stem

            if not self._copy_file_msc(str(generated_video), output_path):
                self.logger.error(f"Failed to copy video to {output_path}")
                return False, None
            self.logger.info(f"Copied generated video to {output_path}")

            if generated_txt.exists():
                output_txt = f"{output_dir}/{output_base}.txt"
                if self._copy_file_msc(str(generated_txt), output_txt):
                    self.logger.info(f"Copied prompt txt to {output_txt}")

            self.logger.debug(f"Cleaned up temp directory: {temp_dir}")
            return True, output_path
