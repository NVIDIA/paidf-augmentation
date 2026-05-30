# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for the augmentation pipeline."""

import os

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tiff",
    ".tif",
}

# Supported video formats
VIDEO_EXTENSIONS = {
    ".mp4",
}

# All supported media formats
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def is_image(file_path: str) -> bool:
    """
    Check if file is an image by extension.

    Args:
        file_path: Path to the file

    Returns:
        bool: True if file extension indicates an image
    """
    return os.path.splitext(file_path)[1].lower() in IMAGE_EXTENSIONS


def is_video(file_path: str) -> bool:
    """
    Check if file is a video by extension.

    Args:
        file_path: Path to the file

    Returns:
        bool: True if file extension indicates a video
    """
    return os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS


def is_media(file_path: str) -> bool:
    """
    Check if file is a supported media file (image or video) by extension.

    Args:
        file_path: Path to the file

    Returns:
        bool: True if file extension indicates a supported media format
    """
    return os.path.splitext(file_path)[1].lower() in MEDIA_EXTENSIONS
