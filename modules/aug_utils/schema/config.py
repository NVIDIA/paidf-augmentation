# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root pipeline configuration model with cross-section validation."""

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from .augmentation import AugmentationConfig, ModelNameEnum, PredictInferenceType
from .captioning import CaptioningConfig
from .data import DataSample
from .data_processing import DataProcessingConfig
from .endpoints import EndpointsConfig
from .evaluators import EvaluatorEntry
from .pipeline import PipelineSettings


class PipelineConfig(BaseModel):
    """Root configuration model for the unified augmentation pipeline.

    Usage::

        import yaml
        from aug_utils.schema import PipelineConfig

        with open("config.yaml") as f:
            raw = yaml.safe_load(f)
        config = PipelineConfig(**raw)
    """

    data: List[DataSample] = Field(min_length=1)
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    captioning: Optional[CaptioningConfig] = Field(default=None)
    augmentation: Optional[AugmentationConfig] = Field(default=None)
    data_processing: Optional[DataProcessingConfig] = Field(default=None)
    evaluators: Optional[List[EvaluatorEntry]] = Field(default=None)

    @model_validator(mode="after")
    def validate_endpoint_requirements(self):
        """Ensure required endpoints exist for the configured modules."""
        # Augmentation -> endpoint mapping
        # Local executors don't need a remote endpoint (they fall back to localhost)
        if self.augmentation is not None:
            from .augmentation import ExecutorTypeEnum

            name = self.augmentation.model.name
            executor = self.augmentation.model.executor_type
            needs_endpoint = executor != ExecutorTypeEnum.LOCAL

            if (
                needs_endpoint
                and name == ModelNameEnum.COSMOS_TRANSFER
                and self.endpoints.cosmos_transfer is None
            ):
                raise ValueError(
                    "augmentation.model.name='cosmos-transfer2.5' with "
                    f"executor_type='{executor.value}' requires "
                    "endpoints.cosmos_transfer to be configured"
                )
            if (
                needs_endpoint
                and name == ModelNameEnum.COSMOS_PREDICT
                and self.endpoints.cosmos_predict is None
            ):
                raise ValueError(
                    "augmentation.model.name='cosmos-predict' with "
                    f"executor_type='{executor.value}' requires "
                    "endpoints.cosmos_predict to be configured"
                )
            if name == ModelNameEnum.IMAGE_EDIT and self.endpoints.image_edit is None:
                raise ValueError(
                    "augmentation.model.name='image-edit' requires "
                    "endpoints.image_edit to be configured"
                )

        # Validate data.inputs.rgb is present when the model requires it
        requires_rgb = True
        if self.augmentation is not None:
            name = self.augmentation.model.name
            inference_type = self.augmentation.parameters.inference_type
            if (
                name == ModelNameEnum.COSMOS_PREDICT
                and inference_type == PredictInferenceType.TEXT2WORLD
            ):
                requires_rgb = False

        if requires_rgb:
            for i, sample in enumerate(self.data):
                inputs = sample.inputs
                rgb = inputs.rgb if inputs is not None else None
                if not rgb:
                    raise ValueError(
                        f"data[{i}].inputs.rgb is required for "
                        f"the configured model (only cosmos-predict "
                        f"with inference_type=text2world allows null inputs)"
                    )

        # Captioning -> endpoint mapping
        if self.captioning is not None:
            if self.captioning.vlm is not None and self.endpoints.vlm is None:
                raise ValueError(
                    "captioning.vlm requires endpoints.vlm to be configured"
                )
            if self.captioning.llm is not None:
                llm = self.captioning.llm
                # text and file_path captioners don't need an LLM endpoint
                needs_llm_endpoint = llm.text is None and llm.file_path is None
                if needs_llm_endpoint and self.endpoints.llm is None:
                    raise ValueError(
                        "captioning.llm requires endpoints.llm to be configured "
                        "(unless 'text' or 'file_path' is set)"
                    )

        return self
