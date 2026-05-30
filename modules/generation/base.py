# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import logging


class BaseGenerator(ABC):
    """Abstract base class for all generation strategies."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @abstractmethod
    def execute(
        self,
        prompt: str,
        input_media_path: str,
        control_inputs: Dict[str, str],
        output_path: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute generation/augmentation.

        Args:
            prompt: Text prompt for generation
            input_media_path: Path to input media (video/image)
            control_inputs: Dict of control inputs (e.g., edge, depth, etc.)
            output_path: Path to save generated output

        Returns:
            Tuple of (success: bool, output_path: str or None)
        """
        pass

    @abstractmethod
    def get_required_inputs(self) -> Dict[str, bool]:
        """
        Return dict of required input types.

        Returns:
            Dict like {"rgb": True, "edge": False, "depth": False}
            True = required, False = optional
        """
        pass
