# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-level settings: retry, logging, and evaluation."""

from typing import Literal

from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Logging configuration."""

    enabled: bool = Field(default=True)
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


class EvaluationSettings(BaseModel):
    """Pipeline-level evaluation behaviour."""

    strict: bool = Field(
        default=True,
        description="If True, fail the sample when any evaluator fails",
    )
    retain_failures: bool = Field(
        default=True,
        description="If True, keep output files even when evaluation fails",
    )


class PipelineSettings(BaseModel):
    """Top-level pipeline settings."""

    retry: int = Field(
        default=1,
        ge=0,
        description="Number of generation retries after evaluator failure",
    )
    regenerate_caption_on_retry: bool = Field(
        default=True,
        description="Regenerate caption from VLM on each retry",
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
