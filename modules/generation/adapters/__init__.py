# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API-contract adapters for the BYOM generation layer."""

from .base import BaseAdapter, Result, resolve_api_key
from .nim import NIMAdapter
from .openai_chat import OpenAIChatAdapter
from .openai_images_edit import OpenAIImageEditAdapter
from .openai_video_sync import OpenAIVideoSyncAdapter
from .openai_video_async import OpenAIVideoAsyncAdapter
from .passthrough import PassthroughAdapter

__all__ = [
    "BaseAdapter",
    "Result",
    "resolve_api_key",
    "OpenAIChatAdapter",
    "OpenAIImageEditAdapter",
    "OpenAIVideoSyncAdapter",
    "OpenAIVideoAsyncAdapter",
    "NIMAdapter",
    "PassthroughAdapter",
]
