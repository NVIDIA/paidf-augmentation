# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known API-contract adapter names for the BYOM generation layer.

This mirrors the adapters registered in :mod:`generation.factory`. It lives in
the schema package (with no generation imports) so cross-section config
validation can reject an unknown ``endpoint.adapter`` without pulling in any
transport SDKs.
"""

KNOWN_ADAPTERS = frozenset(
    {
        "openai.chat.completions",
        "openai.images.edits",
        "openai.video.sync",
        "openai.video.async",
        "nim",
        "passthrough",
    }
)
