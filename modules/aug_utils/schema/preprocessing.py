# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-generation input transforms.

``preprocessing`` is a sub-section of ``data_processing`` (alongside the post-gen
``alignment``). Each sub-field is a transform applied to the input BEFORE
generation; presence-of-field enables it. A transform feeds GENERATION; the
ORIGINAL input is preserved as the reference for steps that compare against the
source (alignment, captioning, hallucination), so e.g. alignment registers the
output back into the true input frame.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ResizeConfig(BaseModel):
    """Resize the input image before generation.

    Useful for routes that edit at the input resolution and have no internal
    upscale lever (e.g. the ``openai.images.edits`` contract): a tiny crop would
    otherwise produce a blobby result. Specify the target as ``target_megapixels``
    (preserve aspect ratio) or explicit ``width`` + ``height``; dimensions snap to
    a multiple of ``grid``.
    """

    enabled: bool = Field(default=True)
    target_megapixels: Optional[float] = Field(
        default=None,
        gt=0,
        description="Resize preserving aspect ratio to ~this many MP. Ignored when width+height are set.",
    )
    width: Optional[int] = Field(
        default=None, gt=0, description="Explicit target width (use with height)."
    )
    height: Optional[int] = Field(
        default=None, gt=0, description="Explicit target height (use with width)."
    )
    grid: int = Field(
        default=32, gt=0, description="Snap target dims to a multiple of this."
    )
    interp: Literal["lanczos", "cubic", "area", "linear", "nearest"] = Field(
        default="lanczos",
        description="Resize interpolation (lanczos best for upscaling).",
    )

    @model_validator(mode="after")
    def _require_a_target(self):
        if (
            self.enabled
            and self.target_megapixels is None
            and not (self.width and self.height)
        ):
            raise ValueError(
                "resize: set 'target_megapixels', or both 'width' and 'height'"
            )
        return self


class PreprocessingConfig(BaseModel):
    """Input transforms applied BEFORE generation. Presence-of-field enables.

    Lives under ``data_processing`` (peer to ``alignment``). A transform feeds the
    generation input; the original input is kept as the reference for alignment.
    """

    resize: Optional[ResizeConfig] = Field(
        default=None,
        description="Resize the input before generation (e.g. upscale a tiny crop).",
    )
