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
- ``transcode``: normalizes the generated video to a single codec (VP9) so the
  output format does not vary with whichever model produced it.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from .preprocessing import PreprocessingConfig


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


class TranscodeParameters(BaseModel):
    """Normalize the generated video to one codec so output format is uniform.

    Augmentation writes whatever bitstream the model endpoint returned, so the
    output codec otherwise varies by model (the Cosmos Transfer NIM emits VP9,
    vLLM-omni emits H.264). Downstream consumers -- and datasets that mix
    augmented clips with UAL artifacts -- expect a single format.

    A source already in the target codec is left byte-identical (no re-encode,
    so no generation loss) unless ``force`` is set. Image outputs are skipped:
    ``data.output.video`` also carries stills for the image-edit model.

    NOTE: transcoding an H.264 source requires a GPU, because the container
    ships only the hardware ``h264_cuvid`` decoder (software AVC decode is off
    for licensing). VP9 sources decode in software and need no GPU. When the
    re-encode cannot run, the original output is kept and the reason is recorded
    in ``data.output.metadata`` under ``transcode.error`` -- the run still
    succeeds, so check that field if a uniform codec is a hard requirement.

    This step runs after the evaluators, so a transcoded artifact is a re-encode
    of the bitstream that was evaluated.
    """

    codec: Literal["vp9"] = Field(
        default="vp9",
        description="Target video codec. VP9 is the only supported target.",
    )
    crf: int = Field(
        default=31,
        ge=0,
        le=63,
        description="libvpx-vp9 constant-quality level (0-63); lower is better quality.",
    )
    speed: int = Field(
        default=4,
        ge=0,
        le=8,
        description=(
            "libvpx-vp9 `-cpu-used` speed/quality tradeoff (0-8); higher is "
            "faster and lower quality."
        ),
    )
    pix_fmt: Literal["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv444p10le"] = (
        Field(
            default="yuv420p",
            description=(
                "Output pixel format. Constrained so a typo (e.g. yuv402p) fails at "
                "config load instead of after generation and evaluation have run."
            ),
        )
    )
    force: bool = Field(
        default=False,
        description=(
            "Re-encode even when the source is already the target codec. Off by "
            "default to avoid a needless generation-loss pass."
        ),
    )


class DataProcessingConfig(BaseModel):
    """Pre- and post-generation processing. Presence-of-field enables each step.

    Lives at the top level of the pipeline config (peer to ``evaluators``).
    ``preprocessing`` runs BEFORE generation (transforms the input);
    ``alignment`` and ``transcode`` run AFTER. ``alignment`` warps the output
    back into the input frame and records recovered parameters in
    ``data.output.metadata``; ``transcode`` normalizes the output codec.

    The two do not interact: ``alignment`` operates on still images, while
    ``transcode`` skips image outputs and only rewrites videos.
    """

    preprocessing: Optional[PreprocessingConfig] = Field(
        default=None,
        description="Input transforms applied BEFORE generation (e.g. resize).",
    )
    alignment: Optional[AlignmentParameters] = Field(
        default=None,
        description="MI-registration alignment of the output back into the input frame",
    )
    transcode: Optional[TranscodeParameters] = Field(
        default=None,
        description="Normalize the generated video to a uniform codec (VP9).",
    )
