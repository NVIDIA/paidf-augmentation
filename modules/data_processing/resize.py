# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-generation input resize (CPU-only, cv2).

Upscales (or downscales) an input image before generation so a route that edits
at the input resolution — e.g. ``openai.images.edits`` — generates at good
detail. Deliberately cv2/numpy only (NO cupy), so resize-only runs work without
a GPU; keep this module free of GPU imports.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import cv2
import multistorageclient as msc
import numpy as np

_INTERP = {
    "lanczos": cv2.INTER_LANCZOS4,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def _snap(value: float, grid: int) -> int:
    """Round ``value`` to the nearest positive multiple of ``grid``."""
    return max(grid, int(round(value / grid)) * grid)


def target_dims(w: int, h: int, config: Dict[str, Any]) -> Tuple[int, int]:
    """Compute (width, height) target from a ResizeConfig dict.

    Explicit ``width``+``height`` win; otherwise derive from
    ``target_megapixels`` preserving the source aspect ratio. Both are snapped to
    a multiple of ``grid``.
    """
    grid = int(config.get("grid") or 32)
    if config.get("width") and config.get("height"):
        return _snap(config["width"], grid), _snap(config["height"], grid)
    mp = float(config["target_megapixels"]) * 1_000_000.0
    ar = w / h
    tw = (mp * ar) ** 0.5
    th = (mp / ar) ** 0.5
    return _snap(tw, grid), _snap(th, grid)


def resize_input(
    src_path: str,
    dest_path: str,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[int, int]:
    """Resize the image at ``src_path`` per ``config`` and write it to ``dest_path``.

    Reads/writes via ``msc`` (local, S3, ...). Drops an alpha channel if present
    (RGBA -> RGB). Returns the (width, height) actually written.
    """
    with msc.open(src_path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"resize: could not decode image at {src_path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # drop alpha (Qwen VAE / cv2 round-trip want 3 channels)

    h, w = img.shape[:2]
    tw, th = target_dims(w, h, config)
    interp = _INTERP.get(str(config.get("interp") or "lanczos"), cv2.INTER_LANCZOS4)
    out = cv2.resize(img, (tw, th), interpolation=interp)

    ok, enc = cv2.imencode(".png", out)
    if not ok:
        raise ValueError("resize: cv2.imencode failed")
    with msc.open(dest_path, "wb") as f:
        f.write(enc.tobytes())

    logger.info(
        "Resize: %dx%d -> %dx%d (%.2f MP, %s) -> %s",
        w,
        h,
        tw,
        th,
        (tw * th) / 1e6,
        config.get("interp") or "lanczos",
        dest_path,
    )
    return tw, th
