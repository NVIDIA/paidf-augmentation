# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pre-generation resize step (cv2-only, no cupy)."""

import logging
import sys

import cv2
import numpy as np
import pytest

from aug_utils.schema.data_processing import DataProcessingConfig
from aug_utils.schema.preprocessing import PreprocessingConfig, ResizeConfig
from data_processing.resize import resize_input, target_dims

LOGGER = logging.getLogger("test")


# ---------------------------------------------------------------------------
# target_dims
# ---------------------------------------------------------------------------
class TestTargetDims:
    def test_explicit_wh_snapped_to_grid(self):
        # 1216x864 already /32; 1200x850 -> nearest /32.
        assert target_dims(158, 114, {"width": 1216, "height": 864, "grid": 32}) == (
            1216,
            864,
        )
        assert target_dims(158, 114, {"width": 1200, "height": 850, "grid": 32}) == (
            1216,
            864,
        )

    def test_megapixels_preserves_aspect_ratio(self):
        tw, th = target_dims(158, 114, {"target_megapixels": 1.05, "grid": 32})
        # ~1.05 MP and aspect ratio close to source (158/114 = 1.386).
        assert 0.9e6 <= tw * th <= 1.2e6
        assert abs((tw / th) - (158 / 114)) < 0.05
        assert tw % 32 == 0 and th % 32 == 0

    def test_grid_default_is_32(self):
        tw, th = target_dims(100, 100, {"target_megapixels": 1.0})
        assert tw % 32 == 0 and th % 32 == 0


# ---------------------------------------------------------------------------
# resize_input
# ---------------------------------------------------------------------------
class TestResizeInput:
    def _write_png(self, path, h, w, channels=3):
        arr = np.full((h, w, channels), 128, dtype=np.uint8)
        cv2.imwrite(str(path), arr)

    def test_upscale_writes_target_dims(self, tmp_path):
        src = tmp_path / "in.png"
        dst = tmp_path / "out.png"
        self._write_png(src, 114, 158)
        tw, th = resize_input(
            str(src), str(dst), {"width": 1216, "height": 864}, LOGGER
        )
        assert (tw, th) == (1216, 864)
        out = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
        assert out.shape[1] == 1216 and out.shape[0] == 864

    def test_rgba_alpha_dropped(self, tmp_path):
        src = tmp_path / "in_rgba.png"
        dst = tmp_path / "out.png"
        self._write_png(src, 64, 64, channels=4)  # RGBA input
        resize_input(str(src), str(dst), {"width": 128, "height": 128}, LOGGER)
        out = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
        assert out.ndim == 3 and out.shape[2] == 3  # alpha dropped -> RGB

    def test_no_cupy_imported(self, tmp_path):
        # The resize path must never import cupy (CPU-only / GPU-less environments).
        # Order-independent: assert resize does not NEWLY import cupy, regardless of
        # whether an earlier test (e.g. alignment) already imported it.
        cupy_before = "cupy" in sys.modules
        src = tmp_path / "in.png"
        dst = tmp_path / "out.png"
        self._write_png(src, 32, 32)
        resize_input(str(src), str(dst), {"target_megapixels": 0.1}, LOGGER)
        assert ("cupy" in sys.modules) == cupy_before


# ---------------------------------------------------------------------------
# ResizeConfig schema
# ---------------------------------------------------------------------------
class TestResizeConfigSchema:
    def test_defaults_and_wh(self):
        c = ResizeConfig(width=1216, height=864)
        assert c.enabled and c.grid == 32 and c.interp == "lanczos"

    def test_megapixels_ok(self):
        assert ResizeConfig(target_megapixels=1.05).target_megapixels == 1.05

    def test_requires_a_target(self):
        with pytest.raises(ValueError, match="target_megapixels"):
            ResizeConfig()  # enabled but no target

    def test_disabled_needs_no_target(self):
        assert ResizeConfig(enabled=False).enabled is False  # no raise

    def test_preprocessing_section_holds_resize(self):
        prep = PreprocessingConfig(resize={"width": 1216, "height": 864})
        assert prep.resize.width == 1216 and prep.resize.height == 864

    def test_data_processing_nests_preprocessing_and_alignment(self):
        dp = DataProcessingConfig(
            preprocessing={"resize": {"width": 1216, "height": 864}},
            alignment={"rot_range_deg": [-1.0, 1.0, 0.1]},
        )
        assert dp.preprocessing.resize.width == 1216
        assert dp.alignment is not None
