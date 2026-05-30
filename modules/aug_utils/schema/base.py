# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared base models for inference parameters."""

from pydantic import BaseModel, Field


class BaseInferenceParameters(BaseModel):
    """Common parameters shared across VLM/LLM inference calls.

    These defaults are tuned for the augmentation pipeline.
    """

    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, gt=0)
    frequency_penalty: float = Field(default=1.05)
    presence_penalty: float = Field(default=0.0)
    stream: bool = Field(default=False)
    retry: int = Field(default=0, ge=0)
