# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
from typing import List, Dict, Optional, Tuple

import multistorageclient as msc


class PassthroughCosmosExecutor:
    """
    Passthrough Cosmos Transfer executor that returns the input without modification.

    This executor can be used for testing or when you want to skip the Cosmos
    transformation step. It simply copies the input video to the output location.

    Note: This is a placeholder/testing executor. For actual Cosmos inference,
    use GradioCosmosExecutor or LocalCosmosExecutor.
    """

    def __init__(
        self,
        endpoint: str,  # Unused for passthrough, kept for interface compatibility
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

        # Default to ct1, or use explicitly configured version
        if model_version is None:
            self.model_version = "ct1"
        else:
            self.model_version = (
                "ct2.5" if model_version.lower() == "ct25" else model_version.lower()
            )

        self.logger.info(
            f"[Passthrough] Initialized Cosmos Transfer version: {self.model_version}"
        )
        self.logger.info(
            f"[Passthrough] Parameters: sigma={sigma}, seed={seed}, guidance={guidance}, steps={num_steps}"
        )

        # TODO: Load Cosmos model here
        self._model = None

    def _load_model(self):
        """
        Load the Cosmos Transfer model into memory.

        TODO: Implement actual model loading based on self.model_version.
        """
        self.logger.info(
            f"[Passthrough] Loading model (version: {self.model_version})..."
        )
        # TODO: Load actual model
        # self._model = CosmosTransferModel.load(self.model_version)
        pass

    def _run_inference(
        self,
        prompt: str,
        input_video_path: str,
        control_videos: Dict[str, str],
    ) -> Optional[bytes]:
        """
        Run Cosmos Transfer inference on the input video.

        Args:
            prompt: Text prompt for generation
            input_video_path: Path to input video
            control_videos: Dictionary of control modality videos

        Returns:
            Generated video bytes or None if failed

        TODO: Implement actual inference logic.
        """
        self.logger.info(
            f"[Passthrough] Running inference with prompt: {prompt[:50]}..."
        )
        self.logger.info(f"[Passthrough] Input video: {input_video_path}")
        self.logger.info(
            f"[Passthrough] Control modalities: {list(control_videos.keys())}"
        )

        # TODO: Run actual inference
        # output = self._model.generate(
        #     prompt=prompt,
        #     input_video=input_video_path,
        #     controls=control_videos,
        #     sigma=self.sigma,
        #     seed=self.seed,
        #     guidance=self.guidance,
        #     num_steps=self.num_steps,
        #     weights=self.weights,
        # )

        # For now, passthrough the input video
        return None

    def execute(
        self,
        prompt: str,
        input_video_path: str,
        control_videos: Dict[str, str],
        output_path: str,
    ) -> Tuple[bool, Optional[str]]:
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
        self.logger.info("[Passthrough] Execute called")
        self.logger.info(f"[Passthrough] Prompt: {prompt}")
        self.logger.info(f"[Passthrough] Input: {input_video_path}")
        self.logger.info(f"[Passthrough] Output: {output_path}")
        self.logger.info(f"[Passthrough] Controls: {control_videos}")

        try:
            # TODO: Replace passthrough with actual inference
            # For now, just copy input video to output (passthrough)
            self.logger.warning(
                "[Passthrough] Using passthrough mode - copying input video to output"
            )

            # Check input video exists
            if not msc.is_file(input_video_path):
                self.logger.error(
                    f"[Passthrough] Input video not found: {input_video_path}"
                )
                return False, None

            # Copy input video to output path
            with (
                msc.open(input_video_path, "rb") as src,
                msc.open(output_path, "wb") as dst,
            ):
                shutil.copyfileobj(src, dst)

            self.logger.info(f"[Passthrough] Successfully saved video to {output_path}")
            return True, output_path

        except Exception as e:
            self.logger.error(f"[Passthrough] Execution failed: {e}")
            return False, None
