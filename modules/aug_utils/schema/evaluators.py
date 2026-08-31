# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluator configuration models.

Evaluators are specified as a list, where each entry contains exactly one of:
- ``hallucination_check``
- ``attribute_verification`` (may contain an inline ``vlm_verification``)
- ``vlm_verification`` (standalone, without attribute verification)
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import BaseInferenceParameters


# ---------------------------------------------------------------------------
# Hallucination Check
# ---------------------------------------------------------------------------


class HallucinationParams(BaseModel):
    """Low-level parameters for motion-artifact hallucination detection."""

    grad_thresh: float = Field(default=10.0)
    blur_ksize: int = Field(default=7)
    morph_k: int = Field(default=3)
    dist_tol_px: float = Field(default=7.0)
    max_frames: Optional[int] = Field(default=None)


class HallucinationCheckEvaluator(BaseModel):
    """Evaluator that detects hallucinated motion artifacts."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(default=True)
    threshold: float = Field(default=0.682, ge=0.0, le=1.0)
    params: HallucinationParams = Field(default_factory=HallucinationParams)


# ---------------------------------------------------------------------------
# VLM Verification
# ---------------------------------------------------------------------------


class VLMVerificationEvaluator(BaseModel):
    """Evaluator that uses a VLM to answer verification questions."""

    executor_type: Optional[Literal["local", "remote"]] = Field(default=None)
    url: Optional[str] = Field(default=None)
    system_prompt: str = Field(default="")
    frames: int = Field(
        default=1,
        ge=1,
        description="Frames to sample from a video, evenly spaced (1 = first frame "
        "only). Use >1 so a mid-video event is visible to the VLM.",
    )
    parameters: BaseInferenceParameters = Field(default_factory=BaseInferenceParameters)
    endpoint_id: Optional[str] = Field(
        default=None,
        description="Pick a specific endpoints[].id; defaults to the single endpoint of the matching role",
    )

    @model_validator(mode="after")
    def validate_remote_executor(self):
        """Ensure url is set when executor_type is remote."""
        if self.executor_type == "remote" and not self.url:
            raise ValueError("url must be specified when executor_type is 'remote'")
        return self


# ---------------------------------------------------------------------------
# Attribute Verification
# ---------------------------------------------------------------------------


class QuestionGenerationConfig(BaseModel):
    """LLM-based MCQ question generation for attribute verification."""

    system_prompt: str = Field(default="")
    generate_options: bool = Field(
        default=False,
        description="Let the LLM invent the distractor options instead of drawing "
        "from verification_options. The correct answer still stays pinned to the "
        "variable's (ground-truth) value.",
    )
    parameters: BaseInferenceParameters = Field(default_factory=BaseInferenceParameters)
    endpoint_id: Optional[str] = Field(
        default=None,
        description="Pick a specific endpoints[].id; defaults to the single endpoint of the matching role",
    )


class ExtraQuestion(BaseModel):
    """A fixed verification question (not LLM-generated)."""

    variable: Optional[str] = Field(
        default=None, description="Variable name this question verifies"
    )
    question: str = Field(description="The question text")
    options: Dict[str, str] = Field(
        description="Answer options keyed by letter (e.g., {'A': '...', 'B': '...'})"
    )
    correct_answer: str = Field(description="Letter of the correct answer")
    request_reasoning: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_correct_answer(self):
        """Ensure correct_answer is a valid key in options."""
        if self.correct_answer not in self.options:
            raise ValueError(
                f"correct_answer '{self.correct_answer}' must be one of "
                f"the option keys: {list(self.options.keys())}"
            )
        return self


class NaturalCaptionConfig(BaseModel):
    """Configuration for generating a natural-language caption after verification passes."""

    system_prompt: Optional[str] = Field(default=None)
    user_prompt_template: Optional[str] = Field(default=None)


class AttributeVerificationEvaluator(BaseModel):
    """Evaluator that checks generated media against target attributes via MCQ."""

    model_config = ConfigDict(extra="forbid")
    executor_type: Optional[Literal["local", "remote"]] = Field(default=None)
    url: Optional[str] = Field(default=None)
    enabled: bool = Field(default=True)
    generate_natural_caption_on_pass: bool = Field(default=False)
    natural_caption: Optional[NaturalCaptionConfig] = Field(default=None)
    exclude_variables: List[str] = Field(default_factory=list)
    extra_questions: Optional[List[ExtraQuestion]] = Field(default=None)
    question_generation: QuestionGenerationConfig = Field(
        default_factory=QuestionGenerationConfig
    )
    vlm_verification: Optional[VLMVerificationEvaluator] = Field(default=None)

    @model_validator(mode="after")
    def validate_remote_executor(self):
        """Ensure url is set when executor_type is remote."""
        if self.executor_type == "remote" and not self.url:
            raise ValueError("url must be specified when executor_type is 'remote'")
        return self


# ---------------------------------------------------------------------------
# Evaluator Entry (list item)
# ---------------------------------------------------------------------------


class EvaluatorEntry(BaseModel):
    """A single entry in the ``evaluators`` list.

    Exactly one of the three evaluator fields must be set.
    """

    hallucination_check: Optional[HallucinationCheckEvaluator] = Field(default=None)
    attribute_verification: Optional[AttributeVerificationEvaluator] = Field(
        default=None
    )
    vlm_verification: Optional[VLMVerificationEvaluator] = Field(default=None)

    @model_validator(mode="after")
    def exactly_one_evaluator(self):
        """Ensure exactly one evaluator type is specified per entry."""
        set_fields = [
            f
            for f in (
                "hallucination_check",
                "attribute_verification",
                "vlm_verification",
            )
            if getattr(self, f) is not None
        ]
        if len(set_fields) != 1:
            raise ValueError(
                f"Each evaluator entry must specify exactly one evaluator type, "
                f"got {len(set_fields)}: {set_fields}"
            )
        return self
