# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captioning configuration models for VLM and LLM captioning strategies."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .base import BaseInferenceParameters


class VLMParameters(BaseInferenceParameters):
    """VLM-specific inference parameters.

    Extends ``BaseInferenceParameters`` with video/image sampling settings.
    """

    fps: float = Field(
        default=4.0, gt=0, description="Frame sampling rate for video input"
    )
    max_pixels: int = Field(
        default=307200, gt=0, description="Maximum pixel count for input frames"
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

    @model_validator(mode="after")
    def _validate_mutual_exclusivity(self) -> "LLMCaptioningConfig":
        if self.text and self.file_path:
            raise ValueError("'text' and 'file_path' are mutually exclusive")
        return self


class CaptioningConfig(BaseModel):
    """Captioning configuration.

    Captioning type is inferred from which sub-sections are present:
    - Only ``vlm``: VLM-only captioning
    - Only ``llm``: LLM-only captioning
    - Both ``vlm`` and ``llm``: VLM+LLM captioning (VLM describes, LLM generates prompt)
    """

    vlm: Optional[VLMCaptioningConfig] = Field(
        default=None,
        description="VLM captioning configuration (describes source media)",
    )
    llm: Optional[LLMCaptioningConfig] = Field(
        default=None,
        description="LLM captioning configuration (generates/polishes prompt)",
    )
