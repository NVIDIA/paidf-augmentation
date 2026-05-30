# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified pipeline configuration schema.

This package defines the Pydantic models that serve as the single source of
truth for the YAML configuration structure used by the augmentation pipeline.
Import ``PipelineConfig`` and call ``PipelineConfig(**yaml_dict)`` to validate
and parse a raw config dictionary.
"""

from .base import BaseInferenceParameters
from .data import ControlInputs, DataInputs, DataOutput, DataSample
from .endpoints import EndpointConfig, EndpointsConfig
from .pipeline import EvaluationSettings, LoggingConfig, PipelineSettings
from .captioning import (
    CaptioningConfig,
    LLMCaptioningConfig,
    LLMParameters,
    VLMCaptioningConfig,
    VLMParameters,
)
from .augmentation import (
    AugmentationConfig,
    AugmentationParameters,
    ExecutorTypeEnum,
    LocalParameters,
    ModalitiesConfig,
    ModelConfig,
    ModelNameEnum,
)
from .data_processing import AlignmentParameters, DataProcessingConfig
from .evaluators import (
    AttributeVerificationEvaluator,
    EvaluatorEntry,
    ExtraQuestion,
    HallucinationCheckEvaluator,
    HallucinationParams,
    NaturalCaptionConfig,
    QuestionGenerationConfig,
    VLMVerificationEvaluator,
)
from .config import PipelineConfig

__all__ = [
    "PipelineConfig",
    "BaseInferenceParameters",
    "ControlInputs",
    "DataInputs",
    "DataOutput",
    "DataSample",
    "EndpointConfig",
    "EndpointsConfig",
    "EvaluationSettings",
    "LoggingConfig",
    "PipelineSettings",
    "CaptioningConfig",
    "LLMCaptioningConfig",
    "LLMParameters",
    "VLMCaptioningConfig",
    "VLMParameters",
    "AugmentationConfig",
    "AugmentationParameters",
    "ExecutorTypeEnum",
    "LocalParameters",
    "ModalitiesConfig",
    "ModelConfig",
    "ModelNameEnum",
    "AlignmentParameters",
    "DataProcessingConfig",
    "AttributeVerificationEvaluator",
    "EvaluatorEntry",
    "ExtraQuestion",
    "HallucinationCheckEvaluator",
    "HallucinationParams",
    "NaturalCaptionConfig",
    "QuestionGenerationConfig",
    "VLMVerificationEvaluator",
]
