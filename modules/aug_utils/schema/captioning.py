# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captioning configuration models for supported prompt-building strategies."""

import string
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .base import BaseInferenceParameters


class VLMParameters(BaseInferenceParameters):
    """VLM-specific inference parameters.

    Extends ``BaseInferenceParameters`` with video/image sampling settings.
    """

    fps: float = Field(
        default=4.0,
        gt=0,
        deprecated=True,
        description=(
            "DEPRECATED / no-op. Was only ever forwarded inside a Qwen2.5-VL-shaped "
            "body; server-side sampling is now declared explicitly under `extra_body` "
            "(e.g. media_io_kwargs.video.fps). Accepted so existing configs keep "
            "validating, but it is not sent to any endpoint."
        ),
    )
    max_pixels: int = Field(
        default=307200,
        gt=0,
        deprecated=True,
        description=(
            "DEPRECATED / no-op. See `fps`; declare the server's own knob under "
            "`extra_body` (e.g. mm_processor_kwargs.videos_kwargs.max_pixels)."
        ),
    )
    extra_body: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Vendor-specific request body forwarded VERBATIM to the VLM chat call "
            "on video input. Omit for servers that reject unknown keys (Gemma's "
            "processor 400s on Qwen's video kwargs). Qwen2.5-VL / NIM example: "
            '{"media_io_kwargs": {"video": {"fps": 4.0}}, "mm_processor_kwargs": '
            '{"videos_kwargs": {"min_pixels": 1568, "max_pixels": 307200}}}'
        ),
    )


class VLMCaptioningConfig(BaseModel):
    """VLM captioning configuration — describes the source media."""

    parser: Literal["instruct", "reasoning"] = Field(
        default="instruct",
        description="VLM output parsing strategy",
    )
    system_prompt: str = Field(default="")
    user_prompt: str = Field(default="")
    parameters: VLMParameters = Field(default_factory=VLMParameters)
    endpoint_id: Optional[str] = Field(
        default=None,
        description="Pick a specific endpoints[].id; defaults to the single endpoint of the matching role",
    )


class LLMParameters(BaseInferenceParameters):
    """LLM-specific inference parameters."""

    pass


class LLMCaptioningConfig(BaseModel):
    """LLM captioning configuration — generates or polishes the final prompt."""

    text: Optional[str] = Field(
        default=None,
        description="Prompt template with {variable_name} placeholders for variable substitution",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path to a prompt file (one prompt per line)",
    )
    system_prompt: Optional[str] = Field(default=None)
    system_prompt_file: Optional[str] = Field(
        default=None,
        description="Path to a file containing the system prompt",
    )
    example_prompt: Optional[str] = Field(
        default=None,
        description="Example prompt shown to the LLM for few-shot guidance",
    )
    parameters: LLMParameters = Field(default_factory=LLMParameters)
    variables: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Variable name to list of possible values for prompt generation",
    )
    verification_values: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Base values used when constructing verification MCQs",
    )
    verification_options: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="All options for MCQ verification questions",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for file captioner shuffling",
    )
    endpoint_id: Optional[str] = Field(
        default=None,
        description="Pick a specific endpoints[].id; defaults to the single endpoint of the matching role",
    )

    @model_validator(mode="after")
    def _validate_mutual_exclusivity(self) -> "LLMCaptioningConfig":
        if self.text and self.file_path:
            raise ValueError("'text' and 'file_path' are mutually exclusive")
        return self


class TemplateAttributeCatalog(BaseModel):
    """Client-defined prompt text for the three supported attribute axes."""

    model_config = ConfigDict(extra="forbid")

    event_type: Dict[str, str] = Field(min_length=1)
    motion_level: Dict[str, str] = Field(min_length=1)
    aftermath: Dict[str, str] = Field(min_length=1)

    @field_validator("event_type", "motion_level", "aftermath")
    @classmethod
    def _normalize_catalog(cls, values: Dict[str, str]) -> Dict[str, str]:
        normalized = {}
        for raw_id, raw_text in values.items():
            value_id = raw_id.strip()
            text = raw_text.strip()
            if not value_id:
                raise ValueError("prompt attribute IDs must not be empty")
            if not text:
                raise ValueError(
                    f"prompt text for attribute ID {value_id!r} must not be empty"
                )
            if value_id in normalized:
                raise ValueError(
                    f"duplicate prompt attribute ID after trimming: {value_id!r}"
                )
            normalized[value_id] = text
        return normalized


class TemplateCaptioningConfig(BaseModel):
    """Deterministic final-prompt rendering from a VLM scene description."""

    model_config = ConfigDict(extra="forbid")

    template: str = Field(
        description=(
            "Final prompt template with scene_caption, event_type_text, "
            "motion_level_text, and aftermath_text placeholders"
        ),
    )
    attributes: TemplateAttributeCatalog = Field(
        description=(
            "Client-defined event_type, motion_level, and aftermath prompt text"
        )
    )

    @model_validator(mode="after")
    def _validate_template(self) -> "TemplateCaptioningConfig":
        if not self.template.strip():
            raise ValueError("captioning.template.template must not be empty")

        try:
            fields = [
                field_name
                for _, field_name, _, _ in string.Formatter().parse(self.template)
                if field_name is not None
            ]
        except ValueError as exc:
            raise ValueError(f"Invalid captioning.template.template: {exc}") from exc

        required = {
            "scene_caption",
            "event_type_text",
            "motion_level_text",
            "aftermath_text",
        }
        unknown = sorted(set(fields) - required)
        if unknown:
            raise ValueError(
                "Unknown captioning.template placeholder(s): " + ", ".join(unknown)
            )
        for field_name in sorted(required):
            if fields.count(field_name) != 1:
                raise ValueError(
                    "captioning.template.template must contain exactly one "
                    f"{{{field_name}}} placeholder"
                )
        return self


class CaptioningConfig(BaseModel):
    """Captioning configuration.

    Captioning type is inferred from which sub-sections are present:
    - Only ``vlm``: VLM-only captioning
    - Only ``llm``: LLM-only captioning
    - Both ``vlm`` and ``llm``: VLM+LLM captioning (VLM describes, LLM generates prompt)
    - Both ``vlm`` and ``template``: VLM+template deterministic prompt building
    """

    vlm: Optional[VLMCaptioningConfig] = Field(
        default=None,
        description="VLM captioning configuration (describes source media)",
    )
    llm: Optional[LLMCaptioningConfig] = Field(
        default=None,
        description="LLM captioning configuration (generates/polishes prompt)",
    )
    template: Optional[TemplateCaptioningConfig] = Field(
        default=None,
        description="Deterministic prompt rendering from a VLM scene description",
    )

    @model_validator(mode="after")
    def _validate_template_strategy(self) -> "CaptioningConfig":
        if self.template is not None and self.vlm is None:
            raise ValueError("captioning.template requires captioning.vlm")
        if self.template is not None and self.llm is not None:
            raise ValueError(
                "captioning.template cannot be combined with captioning.llm"
            )
        return self
