# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passthrough (echo) adapter — a testing/debug seam, no inference.

Replaces the old ``PassthroughCosmosExecutor``: it echoes the input media back
as the output (or the prompt as text when no media is present), so the full
config → factory → executor → write path can be exercised without a live
endpoint.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseAdapter, Result, first_media


class PassthroughAdapter(BaseAdapter):
    """Returns the input unchanged. Useful for tests and dry runs."""

    def invoke(self, payload: Dict[str, Any]) -> Result:
        # No wire body exists on this contract, so record the normalized payload
        # itself; a dry run still reports what it would have sent.
        self.record_request(
            {"prompt": payload.get("prompt"), **(payload.get("params") or {})}
        )
        media = first_media(payload)
        if media is not None:
            self.logger.info("PassthroughAdapter: echoing input media")
            return Result(media_bytes=media["bytes"], request_id="passthrough")
        prompt = payload.get("prompt")
        self.logger.info("PassthroughAdapter: echoing prompt text")
        return Result(text=prompt or "", request_id="passthrough")
