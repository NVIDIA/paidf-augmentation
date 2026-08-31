# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic prompt construction from a VLM scene description."""

import json
import logging
from typing import Any, Dict, Optional

from captioning.base import BaseCaptioner


class TemplateCaptioner(BaseCaptioner):
    """Combine factual VLM output with three selected prompt attributes."""

    ATTRIBUTE_NAMES = ("event_type", "motion_level", "aftermath")

    def __init__(
        self,
        template: str,
        attributes: Dict[str, Dict[str, str]],
        vlm_captioner: BaseCaptioner,
        logger: logging.Logger,
    ):
        super().__init__(logger)
        if vlm_captioner is None:
            raise ValueError("VLM-template captioning requires a VLM captioner")
        if not isinstance(template, str) or not template.strip():
            raise ValueError("captioning.template.template must not be empty")
        if not isinstance(attributes, dict) or set(attributes) != set(
            self.ATTRIBUTE_NAMES
        ):
            raise ValueError(
                "captioning.template.attributes must define exactly event_type, "
                "motion_level, and aftermath"
            )

        self.template = template
        self.attributes = {
            name: {
                value_id.strip(): prompt_text.strip()
                for value_id, prompt_text in catalog.items()
            }
            for name, catalog in attributes.items()
        }
        self.vlm_captioner = vlm_captioner
        self._reset_last_state()

    def _reset_last_state(self) -> None:
        self.last_prompt_builder: Optional[str] = None
        self.last_scene_description: Optional[str] = None
        self.last_scene_description_source: Optional[str] = None

    def reset_provenance(self) -> None:
        """Clear per-sample provenance when the CLI reuses a cached prompt."""
        self._reset_last_state()

    @staticmethod
    def _sample_inputs(sample: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(sample, dict):
            return {}
        inputs = sample.get("inputs") or {}
        return inputs if isinstance(inputs, dict) else {}

    @classmethod
    def _source_uri(
        cls, sample: Optional[Dict[str, Any]], media_path: Optional[str]
    ) -> str:
        source_uri = media_path or cls._sample_inputs(sample).get("rgb")
        if not isinstance(source_uri, str) or not source_uri.strip():
            raise ValueError(
                "VLM-template captioning requires a source image or video in "
                "data.inputs.rgb"
            )
        return source_uri.strip()

    def _resolve_prompt_attributes(
        self, sample: Optional[Dict[str, Any]]
    ) -> tuple[Dict[str, str], Dict[str, str]]:
        selections = self._sample_inputs(sample).get("prompt_attributes")
        if hasattr(selections, "model_dump"):
            selections = selections.model_dump()
        if not isinstance(selections, dict):
            raise ValueError(
                "VLM-template captioning requires data.inputs.prompt_attributes"
            )

        try:
            selected_ids = {
                name: selections[name].strip() for name in self.ATTRIBUTE_NAMES
            }
            selected_text = {
                f"{name}_text": self.attributes[name][selected_ids[name]]
                for name in self.ATTRIBUTE_NAMES
            }
        except (AttributeError, KeyError) as exc:
            raise ValueError(
                "data.inputs.prompt_attributes must select a defined event_type, "
                "motion_level, and aftermath"
            ) from exc
        return selected_ids, selected_text

    @staticmethod
    def _validate_scene_caption(caption: Any) -> str:
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("VLM returned an empty scene description")

        text = caption.strip()
        if "```" in text:
            raise ValueError(
                "VLM scene description must be plain text, not code-fenced"
            )
        # Only an object/array is really "JSON instead of prose". A quoted or
        # numeric caption also parses (json.loads('"a street."') -> str), and
        # instruct models routinely wrap output in quotes -- rejecting those
        # discarded a perfectly good caption and killed the sample.
        if text[0] in "{[":
            try:
                json.loads(text)
            except json.JSONDecodeError:
                return text
            raise ValueError("VLM scene description must be plain text, not JSON")
        return text

    def _render_prompt(self, values: Dict[str, str]) -> str:
        try:
            prompt = self.template.format_map(values).strip()
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Failed to render captioning.template.template: {exc}"
            ) from exc
        if not prompt:
            raise ValueError("Rendered VLM-template prompt is empty")
        return prompt

    def get_caption_for_sample(
        self, sample: Dict[str, Any], media_path: Optional[str]
    ) -> str:
        """Build one VLM-grounded prompt for the sample's selected IDs."""
        self._reset_last_state()
        source_uri = self._source_uri(sample, media_path)
        selected_ids, selected_text = self._resolve_prompt_attributes(sample)
        scene_caption = self._validate_scene_caption(
            self.vlm_captioner.get_caption(source_uri)
        )
        prompt = self._render_prompt({"scene_caption": scene_caption, **selected_text})

        self.last_scene_description = scene_caption
        self.last_scene_description_source = "vlm"
        self.last_prompt_builder = "vlm_template"
        self.logger.info(
            "Built VLM-template prompt (event_type=%s, motion_level=%s, aftermath=%s)",
            selected_ids["event_type"],
            selected_ids["motion_level"],
            selected_ids["aftermath"],
        )
        return prompt

    def get_caption(self, media_path: str) -> str:
        """Reject calls that omit the required per-sample selections."""
        raise ValueError(
            "TemplateCaptioner requires data.inputs.prompt_attributes; "
            "use get_caption_for_sample"
        )
