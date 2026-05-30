# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import string
from typing import Any, Dict, Optional
from captioning.base import BaseCaptioner


class TextCaptioner(BaseCaptioner):
    """
    Fixed text captioning — returns the same text for all media.

    Supports variable substitution: if ``variables`` are provided, Python
    ``str.format_map`` is applied to the template so that placeholders like
    ``{top_outer_color}`` are replaced with the first value from each
    variable list.
    """

    def __init__(
        self,
        text: str,
        logger: logging.Logger,
        variables: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(logger)
        self.template = text
        self.variables = self._normalize_variables(variables or {})
        # Validate placeholders match variable keys
        placeholders = {
            name for _, name, _, _ in string.Formatter().parse(text) if name is not None
        }
        if placeholders and self.variables:
            missing = placeholders - set(self.variables.keys())
            if missing:
                raise ValueError(
                    f"Template has placeholders {missing} not found in variables. "
                    f"Available variables: {set(self.variables.keys())}"
                )
            unused = set(self.variables.keys()) - placeholders
            if unused:
                logger.warning(
                    f"Variables {unused} are defined but not used in template"
                )
        elif placeholders and not self.variables:
            raise ValueError(
                f"Template has placeholders {placeholders} but no variables provided"
            )
        # Pre-render the prompt
        if self.variables:
            self.text = self.template.format_map(self.variables)
            self.logger.info(
                f"Text captioner initialized (with variable substitution): {self.text[:100]}..."
            )
        else:
            self.text = self.template
            self.logger.info(f"Text captioner initialized: {self.text[:100]}...")

    @staticmethod
    def _normalize_variables(variables: Dict[str, Any]) -> Dict[str, str]:
        normalized = {}
        for key, value in variables.items():
            if isinstance(value, list):
                if value:
                    normalized[key] = str(value[0])
            elif value is not None:
                normalized[key] = str(value)
        return normalized

    def get_caption(self, media_path: str) -> str:
        """Return the (substituted) text. media_path is unused."""
        self.logger.debug(f"Using text caption for {media_path}")
        return self.text
