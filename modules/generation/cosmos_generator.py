# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, List, Tuple, Optional

from .base import BaseGenerator
from .cosmos.gradio_executor import GradioCosmosExecutor
from .cosmos.passthrough_executor import PassthroughCosmosExecutor


class CosmosGenerator(BaseGenerator):
    """COSMOS Transfer model generator."""

    def __init__(
        self,
        executor_type: str,
        endpoint: str,
        sigma: int,
        seed: int,
        guidance: int,
        num_steps: int,
        modalities: List[str],
        weights: Dict[str, float],
        positive_prompt: str,
        negative_prompt: str,
        inference_name: str,
        logger: logging.Logger,
        model_version: Optional[str] = None,
        **kwargs,  # Allow extra params for future extensibility
    ):
        super().__init__(logger)

        self.executor_type = executor_type
        self.modalities = modalities

        # Initialize appropriate executor
        if executor_type == "gradio":
            self.executor = GradioCosmosExecutor(
                endpoint=endpoint,
                sigma=sigma,
                seed=seed,
                guidance=guidance,
                num_steps=num_steps,
                modalities=modalities,
                weights=weights,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                logger=logger,
                inference_name=inference_name,
                model_version=model_version,
                **kwargs,
            )
        elif executor_type == "local":
            from .cosmos.local_executor import LocalCosmosExecutor

            self.executor = LocalCosmosExecutor(
                sigma=sigma,
                seed=seed,
                guidance=guidance,
                num_steps=num_steps,
                modalities=modalities,
                weights=weights,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                logger=logger,
                inference_name=inference_name,
                model_version=model_version,
                **kwargs,
            )
        elif executor_type == "passthrough":
            self.executor = PassthroughCosmosExecutor(
                endpoint=endpoint,
                sigma=sigma,
                seed=seed,
                guidance=guidance,
                num_steps=num_steps,
                modalities=modalities,
                weights=weights,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                logger=logger,
                inference_name=inference_name,
                model_version=model_version,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown COSMOS executor type: {executor_type}. "
                f"Valid types: gradio, local, passthrough"
            )

        self.logger.info(f"COSMOS generator initialized with {executor_type} executor")

    def execute(
        self,
        prompt: str,
        input_media_path: str,
        control_inputs: Dict[str, str],
        output_path: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute COSMOS Transfer generation.

        Args:
            prompt: Text prompt for generation
            input_media_path: Path to input video/image
            control_inputs: Dict of control videos (edge, depth, seg, vis)
            output_path: Path to save generated video/image

        Returns:
            Tuple of (success: bool, output_path: str or None)
        """
        # Delegate to executor (uses control_videos naming internally)
        return self.executor.execute(
            prompt=prompt,
            input_video_path=input_media_path,
            control_videos=control_inputs,
            output_path=output_path,
        )

    def get_required_inputs(self) -> Dict[str, bool]:
        """
        Return COSMOS input requirements.

        Returns:
            Dict indicating RGB is required, control modalities are optional
        """
        requirements = {"rgb": True}  # RGB input always required

        # Control modalities are optional
        for modality in ["edge", "depth", "seg", "vis"]:
            requirements[modality] = False

        return requirements
