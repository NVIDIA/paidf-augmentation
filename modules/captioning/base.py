# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import logging


class BaseCaptioner(ABC):
    """Abstract base class for all captioning strategies."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @abstractmethod
    def get_caption(self, media_path: str) -> str:
        """
        Generate or retrieve caption for media.

        Args:
            media_path: Path to media file (may be unused for non-media strategies)

        Returns:
            str: Generated caption
        """
        pass
