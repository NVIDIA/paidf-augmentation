# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint configuration models.

The BYOM refactor replaces the old fixed ``EndpointsConfig`` mapping (one
optional slot per role) with a flat *list* of :class:`Endpoint` entries. Each
entry declares its ``role`` and an optional API-contract ``adapter``, so the
pipeline can host arbitrary "bring your own model" endpoints.
"""

from typing import Optional

from pydantic import BaseModel, Field


class Endpoint(BaseModel):
    """Configuration for a single API endpoint in the endpoint registry."""

    id: Optional[str] = Field(
        default=None,
        description="Optional unique handle; only needed to disambiguate 2+ endpoints sharing a role",
    )
    role: str = Field(
        description="Endpoint role (vlm, llm, image_edit, video_transfer, ...)"
    )
    url: str = Field(description="Endpoint URL")
    model: str = Field(default="", description="Model name or identifier")
    adapter: Optional[str] = Field(
        default=None,
        description="API-contract adapter (defaults to the role's default)",
    )
    api_key_env: Optional[str] = Field(
        default=None,
        description=(
            "Env var holding the API key. Keys must come from the environment, "
            "not a config literal, so they never leak through CLI overrides/logs."
        ),
    )
    timeout: Optional[float] = Field(
        default=None,
        description=(
            "HTTP timeout in seconds; also the total invocation and polling "
            "budget for asynchronous NVCF NIM requests"
        ),
    )
