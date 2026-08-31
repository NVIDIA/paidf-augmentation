# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Augmentation configuration models (replaces the old ``generation`` section)."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelNameEnum(str, Enum):
    """Historically supported augmentation model names.

    With BYOM, ``ModelConfig.name`` is a free-form string and is no longer
    constrained to this enum; the enum is retained because other code/tests
    still reference its canonical values.
    """

    COSMOS_TRANSFER = "cosmos-transfer2.5"
    COSMOS_PREDICT = "cosmos-predict"
    IMAGE_EDIT = "image-edit"


class PredictInferenceType(str, Enum):
    """Cosmos Predict 2.5 inference types."""

    TEXT2WORLD = "text2world"
    IMAGE2WORLD = "image2world"
    VIDEO2WORLD = "video2world"


class ModelConfig(BaseModel):
    """Augmentation model selection (BYOM: free-form model name)."""

    name: str = Field(description="Model to use for augmentation")
    version: Optional[str] = Field(default=None, description="Model version (TBD)")


class AugmentationParameters(BaseModel):
    """Union of all model-specific generation parameters.

    BYOM-friendly by design: this is the pass-through surface to whatever model
    the config selects, so it is intentionally permissive.
      * ``extra="allow"`` — params NOT listed below (a new model's own knobs) are
        accepted and forwarded to the wire as-is. Adding a model needs no schema
        change.
      * ``coerce_numbers_to_str=True`` — a value written as a number for a
        string-typed field is coerced rather than rejected (e.g. ``resolution:
        720`` -> ``"720"``, which is what the Cosmos NIM expects on the wire). So
        a config can write the knob the natural way regardless of the field's
        declared type.
    Only the fields relevant to the selected model are used at runtime.
    """

    model_config = ConfigDict(extra="allow", coerce_numbers_to_str=True)

    # Shared parameters
    seed: Optional[int] = Field(default=None, description="Random seed (None = auto)")
    guidance: Optional[float] = Field(
        default=3.0, description="Classifier-free guidance scale"
    )
    num_steps: Optional[int] = Field(
        default=35, gt=0, description="Number of inference steps"
    )

    # Cosmos Transfer 2.5 specific
    sigma: Optional[int] = Field(
        default=90, description="Noise sigma for Cosmos Transfer"
    )
    inference_name: Optional[str] = Field(
        default=None, description="Cosmos inference entry point name"
    )

    # Cosmos Predict 2.5 specific
    inference_type: Optional[PredictInferenceType] = Field(
        default=None,
        description="Predict inference type: text2world, image2world, or video2world",
    )
    num_output_frames: Optional[int] = Field(
        default=None, gt=0, description="Number of output frames to generate"
    )
    enable_autoregressive: Optional[bool] = Field(
        default=None, description="Enable autoregressive generation"
    )
    chunk_size: Optional[int] = Field(
        default=None, gt=0, description="Chunk size for autoregressive generation"
    )
    chunk_overlap: Optional[int] = Field(
        default=None, ge=0, description="Overlap between chunks"
    )
    resolution: Optional[str] = Field(
        default=None, description="Resolution override for Cosmos Predict"
    )
    offload_tokenizer: Optional[bool] = Field(
        default=None,
        description="Offload tokenizer to CPU between steps to save GPU memory",
    )
    offload_text_encoder: Optional[bool] = Field(
        default=None,
        description="Offload text encoder to CPU after embedding computation",
    )

    # Image edit specific
    num_inference_steps: Optional[int] = Field(
        default=None, gt=0, description="Image edit inference step count"
    )
    guidance_scale: Optional[float] = Field(
        default=None, description="Image edit guidance scale"
    )
    negative_prompt: Optional[str] = Field(
        default=" ",
        description=(
            "Image edit negative prompt for classifier-free guidance. "
            "Image-Edit-Plus only enables CFG when negative_prompt is not "
            "None; ' ' (single space) is the canonical sentinel that turns on "
            "CFG without a real negative concept. Set to null to disable CFG."
        ),
    )


class ModalitiesConfig(BaseModel):
    """Control modality weights and prompts for Cosmos Transfer 2.5.

    Numeric fields are the modality weights; string fields are generation prompts.
    """

    edge: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    depth: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    seg: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    vis: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    seg_control_prompt: Optional[str] = Field(
        default=None,
        description="Segmentation class names for seg modality control",
    )
    positive_prompt: Optional[str] = Field(default=None)
    negative_prompt: Optional[str] = Field(default=None)


class AugmentationConfig(BaseModel):
    """Top-level augmentation configuration (replaces ``generation``)."""

    model: ModelConfig
    parameters: AugmentationParameters = Field(default_factory=AugmentationParameters)
    modalities: Optional[ModalitiesConfig] = Field(
        default=None,
        description="Control modalities (only for cosmos-transfer2.5)",
    )
