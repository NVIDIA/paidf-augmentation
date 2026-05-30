# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import random
from typing import Optional

import multistorageclient as msc

from captioning.base import BaseCaptioner


class FileCaptioner(BaseCaptioner):
    """Random selection from a file containing captions (one per line)."""

    def __init__(
        self, file_path: str, logger: logging.Logger, seed: Optional[int] = None
    ):
        super().__init__(logger)
        self.file_path = file_path
        self.captions = []
        self._load_captions()

        if seed is not None:
            random.seed(seed)
            self.logger.info(f"File captioner using seed: {seed}")

    def _load_captions(self):
        """Load captions from file."""
        if not msc.is_file(self.file_path):
            raise FileNotFoundError(f"Caption file not found: {self.file_path}")

        with msc.open(self.file_path, "r") as f:
            self.captions = [line.strip() for line in f if line.strip()]

        if not self.captions:
            raise ValueError(f"No captions found in file: {self.file_path}")

        self.logger.info(f"Loaded {len(self.captions)} captions from {self.file_path}")

    def get_caption(self, media_path: str) -> str:
        """Return random caption from file."""
        caption = random.choice(self.captions)
        self.logger.debug(f"Selected caption: {caption[:100]}...")
        return caption
