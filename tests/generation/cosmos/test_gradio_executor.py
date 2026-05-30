# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `generation.cosmos.gradio_executor`."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from generation.cosmos.gradio_executor import GradioCosmosExecutor


class TestGradioCosmosExecutor:
    """Tests the `GradioCosmosExecutor` class."""

    @pytest.fixture
    def gradio_executor(self) -> GradioCosmosExecutor:
        mock_client = MagicMock()
        with patch("generation.cosmos.gradio_executor.Client", mock_client):
            return GradioCosmosExecutor(
                endpoint="http://localhost:30002/",
                sigma=10,
                seed=42,
                guidance=7.5,
                num_steps=50,
                modalities=["edge", "depth", "seg", "vis"],
                weights={"edge": 1.0, "depth": 1.0, "seg": 1.0, "vis": 1.0},
                positive_prompt="test positive prompt",
                negative_prompt="test negative prompt",
                logger=logging.getLogger("test"),
                inference_name="test",
                model_version="ct2.5",
            )

    def test_build_ct25_params(self, gradio_executor: GradioCosmosExecutor) -> None:
        """Test the build_ct25_params method."""
        params = gradio_executor._build_ct25_params(
            prompt_path="prompt.txt",
            video="video.mp4",
            controls={
                "edge": None,
                "depth": "depth.mp4",
                "seg": "seg.mp4",
                "vis": "vis.mp4",
            },
            available={
                "edge": None,
                "depth": "depth.mp4",
                "seg": "seg.mp4",
                "vis": "vis.mp4",
            },
        )
        assert params is not None
        logging.info("params: %s", params)
        assert params["name"] == "test"
        assert params["prompt_path"] == "prompt.txt"
        assert params["video_path"] == "video.mp4"
        assert params["guidance"] == 7.5
        assert params["num_steps"] == 50
        assert params["seed"] == 42
        assert params["sigma_max"] == "10"
        assert params["edge"]["control_weight"] == 1.0
        assert params["depth"]["control_path"] == "depth.mp4"
        assert params["depth"]["control_weight"] == 1.0
        assert params["seg"]["control_path"] == "seg.mp4"
        assert params["seg"]["control_weight"] == 1.0
        assert params["vis"]["control_path"] == "vis.mp4"

    def test_build_ct25_params_optional_controls(
        self, gradio_executor: GradioCosmosExecutor
    ) -> None:
        """Test the build_ct25_params method with missing optional control paths."""
        params = gradio_executor._build_ct25_params(
            prompt_path="prompt.txt",
            video="video.mp4",
            controls={
                "edge": None,
                "depth": "depth.mp4",
                "seg": None,
                "vis": "vis.mp4",
            },
            available={
                "edge": None,
                "depth": "depth.mp4",
                "seg": None,
                "vis": "vis.mp4",
            },
        )

        assert params is not None
        assert params["seg"]["control_weight"] == 1.0
        assert "control_path" not in params["seg"]
