#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from captioning.base import BaseCaptioner
from captioning.llm_captioner import LLMCaptioner
from captioning.vlm_captioner import VLMCaptioner


class VLMLLMCaptioner(BaseCaptioner):
    """Compose VLM scene captioning with LLM prompt synthesis."""

    def __init__(
        self,
        vlm_captioner: VLMCaptioner,
        llm_captioner: LLMCaptioner,
        logger: logging.Logger,
    ):
        super().__init__(logger)
        self.vlm_captioner = vlm_captioner
        self.llm_captioner = llm_captioner

    def get_caption(self, media_path: str) -> str:
        self.logger.info("Generating intermediate caption with VLM")
        source_caption = self.vlm_captioner.get_caption(media_path)
        self.logger.debug("VLM source caption: %s", source_caption)

        self.logger.info("Generating final prompt with LLM")
        final_prompt = self.llm_captioner.generate_prompt(
            media_path=media_path,
            source_caption=source_caption,
        )
        self.logger.debug("VLM+LLM final prompt: %s", final_prompt)
        return final_prompt
