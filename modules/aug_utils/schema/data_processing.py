# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level post-processing configuration for augmentation outputs.

``data_processing`` is a peer of ``augmentation`` and ``evaluators`` in the
pipeline config. Each sub-field is a separate post-processor; presence of the
sub-field enables it.

Currently supported:
- ``alignment``: 5-DOF affine MI-registration that warps a generated output
  back into the reference frame of the input image. Mirrors the parameters of
  the standalone tool at ``data/alignment_module/_registration.py``.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field


RangeTriple = Annotated[
    list[float],
    Field(min_length=3, max_length=3, description="[min, max, step]"),
]


class AlignmentParameters(BaseModel):
    """MI registration grid-search and refinement knobs.

    Most fields have sensible auto-derived defaults — do NOT override unless
    alignment is visibly failing. See ``run_alignment`` for the derivation
    rules and the YAML config for guidance.
    """

    sx_range: Optional[RangeTriple] = Field(
        default=None,
        description=(
            "scale-X search [min, max, step]; leave null — auto-derived from "
            "generated/ref width ratio (±10%). Override only if alignment is "
            "visibly failing for an unusual generator."
        ),
    )
    sy_range: Optional[RangeTriple] = Field(
        default=None,
        description=(
            "scale-Y search [min, max, step]; leave null — auto-derived from "
            "generated/ref height ratio (±10%). Override only if alignment is "
            "visibly failing for an unusual generator."
        ),
    )
    rot_range_deg: RangeTriple = Field(
        default=[-1.0, 1.0, 0.1], description="rotation search degrees [min, max, step]"
    )
    shift_range: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "tx/ty half-range (px); leave null — auto-derived as "
            "max(ref_h, ref_w) // 4. Override only if alignment is visibly failing."
        ),
    )
    shift_step: int = Field(default=1, gt=0, description="tx/ty grid step (px)")
    bins: int = Field(default=64, gt=0, description="MI histogram bins")
    pyr_levels: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "image pyramid levels; leave null — auto-derived from ref dims "
            "(smallest level kept ≥32px, capped at 3). "
            "Override only if alignment is visibly failing."
        ),
    )
    interp: Literal["nearest", "bilinear"] = Field(
        default="nearest", description="warp interpolation kernel"
    )
    no_resize: bool = Field(default=True, description="skip pre-resize of align to ref")
    min_mi: Optional[float] = Field(
        default=None, description="abort threshold; null = no gate"
    )


class DataProcessingConfig(BaseModel):
    """Post-processors applied after augmentation. Presence-of-field enables.

    Lives at the top level of the pipeline config (peer to ``evaluators``).
    Each post-processor mutates the augmentation output in place — the aligned
    image overwrites ``data.output.video``; recovered parameters are recorded
    in ``data.output.metadata`` under the matching sub-key (e.g. ``alignment``).
    """

    alignment: Optional[AlignmentParameters] = Field(
        default=None,
        description="MI-registration alignment of the output back into the input frame",
    )
