# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint configuration models."""

from typing import Optional

from pydantic import BaseModel, Field


class EndpointConfig(BaseModel):
    """Configuration for a single API endpoint."""

    url: str = Field(description="Endpoint URL")
    model: str = Field(default="", description="Model name or identifier")


class EndpointsConfig(BaseModel):
    """All available API endpoints for the pipeline."""

    vlm: Optional[EndpointConfig] = Field(
        default=None, description="Vision Language Model endpoint"
    )
    llm: Optional[EndpointConfig] = Field(
        default=None, description="Large Language Model endpoint"
    )
    cosmos_transfer: Optional[EndpointConfig] = Field(
        default=None, description="Cosmos Transfer 2.5 endpoint"
    )
    cosmos_predict: Optional[EndpointConfig] = Field(
        default=None, description="Cosmos Predict 2.5 endpoint"
    )
    image_edit: Optional[EndpointConfig] = Field(
        default=None, description="Image edit endpoint"
    )
