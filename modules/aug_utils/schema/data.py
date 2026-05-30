# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data sample models for pipeline input/output specification."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ControlInputs(BaseModel):
    """Optional control modality paths (used only for Cosmos Transfer 2.5)."""

    edge: Optional[str] = Field(default=None, description="Edge control video path")
    depth: Optional[str] = Field(default=None, description="Depth control video path")
    seg: Optional[str] = Field(
        default=None, description="Segmentation control video path"
    )
    vis: Optional[str] = Field(
        default=None, description="Visibility control video path"
    )


class DataInputs(BaseModel):
    """Input specification for a single data sample."""

    rgb: Optional[str] = Field(
        default=None,
        description="Path to the input RGB video or image (not required for text2world)",
    )
    controls: Optional[ControlInputs] = Field(
        default=None,
        description="Control modality inputs (only for Cosmos Transfer 2.5)",
    )


class DataOutput(BaseModel):
    """Output specification for a single data sample."""

    video: str = Field(description="Path for the output video or image")
    caption: str = Field(description="Path for the generated caption/prompt text file")
    metadata: str = Field(description="Path for the output metadata JSON file")
    evaluation: Optional[str] = Field(
        default=None,
        description="Path for the evaluation results JSON file",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_media_as_video(cls, values):
        """Accept ``output.media`` as an alias for ``output.video``."""
        if isinstance(values, dict) and "media" in values and "video" not in values:
            values["video"] = values.pop("media")
        return values


class DataSample(BaseModel):
    """A single input/output sample to process."""

    inputs: Optional[DataInputs] = Field(
        default=None,
        description="Input specification (null for text2world with no input media)",
    )
    output: DataOutput
