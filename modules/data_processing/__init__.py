# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Processing for augmentation I/O (pre-gen resize, post-gen alignment, ...)."""

__all__ = ["run_alignment", "resize_input", "transcode_video"]


def __getattr__(name):
    # Lazy: importing ``run_alignment`` pulls in cupy (GPU-only). Keep it out of
    # package import so cv2-only consumers (e.g. ``resize``) don't require a GPU.
    if name == "run_alignment":
        from .alignment import run_alignment

        return run_alignment
    if name == "resize_input":
        from .resize import resize_input

        return resize_input
    if name == "transcode_video":
        from .transcode import transcode_video

        return transcode_video
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
