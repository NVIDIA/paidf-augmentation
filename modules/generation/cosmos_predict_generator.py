# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos Predict 2.5 generator — delegates to a local torchrun executor."""

import logging
from typing import Dict, Optional, Tuple

from .base import BaseGenerator
from .cosmos.local_predict_executor import LocalCosmosPredictExecutor


class CosmosPredictGenerator(BaseGenerator):
    """Cosmos Predict 2.5 generator.

    Supports three inference types:
    - ``text2world``: Generate video from text prompt only.
    - ``image2world``: Generate video from a single image + prompt.
    - ``video2world``: Extend or modify existing video + prompt.
    """

    def __init__(
        self,
        endpoint: str,
        inference_type: str,
        num_output_frames: Optional[int],
        num_steps: int,
        enable_autoregressive: Optional[bool],
        chunk_size: Optional[int],
        chunk_overlap: Optional[int],
        seed: int,
        guidance: float,
        inference_name: str,
        resolution: str,
        logger: logging.Logger,
        **kwargs,
    ):
        super().__init__(logger=logger)
        self.inference_type = inference_type

        self.executor = LocalCosmosPredictExecutor(
            inference_type=inference_type,
            seed=seed,
            guidance=guidance,
            num_steps=num_steps,
            num_output_frames=num_output_frames,
            enable_autoregressive=enable_autoregressive,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            resolution=resolution,
            inference_name=inference_name,
            logger=logger,
            **kwargs,
        )

    @property
    def seed(self) -> int:
        return self.executor.seed

    @seed.setter
    def seed(self, value: int) -> None:
        self.executor.seed = value

    def execute(
        self,
        prompt: str,
        input_media_path: Optional[str],
        control_inputs: Dict[str, str],
        output_path: str,
    ) -> Tuple[bool, Optional[str]]:
        """Execute Cosmos Predict video generation.

        ``control_inputs`` is accepted for interface compatibility but ignored
        — Cosmos Predict does not use control modalities.
        """
        return self.executor.execute(prompt, input_media_path, output_path)

    def get_required_inputs(self) -> Dict[str, bool]:
        """Return input requirements based on inference type."""
        if self.inference_type == "text2world":
            return {"rgb": False}
        return {"rgb": True}
