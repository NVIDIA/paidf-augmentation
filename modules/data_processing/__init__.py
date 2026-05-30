# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-processing for augmentation outputs (alignment, ...)."""

from .alignment import run_alignment

__all__ = ["run_alignment"]
