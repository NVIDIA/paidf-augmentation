# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CosmosPredictGenerator."""

import sys
from unittest.mock import MagicMock

# Mock the local_predict_executor module before importing the generator
_mock_executor_module = MagicMock()
sys.modules["generation.cosmos.local_predict_executor"] = _mock_executor_module
sys.modules.setdefault("cosmos_oss", MagicMock())
sys.modules.setdefault("cosmos_oss.init", MagicMock())

from generation.cosmos_predict_generator import CosmosPredictGenerator  # noqa: E402


def _make_generator(inference_type="video2world", **kwargs):
    """Create a CosmosPredictGenerator with mocked executor."""
    return CosmosPredictGenerator(
        endpoint="http://localhost:30003/",
        inference_type=inference_type,
        num_output_frames=kwargs.get("num_output_frames", 200),
        num_steps=kwargs.get("num_steps", 35),
        enable_autoregressive=kwargs.get("enable_autoregressive"),
        chunk_size=kwargs.get("chunk_size"),
        chunk_overlap=kwargs.get("chunk_overlap"),
        seed=kwargs.get("seed", 42),
        guidance=kwargs.get("guidance", 3.0),
        inference_name=kwargs.get("inference_name", "test_predict"),
        resolution=kwargs.get("resolution", "none"),
        logger=MagicMock(),
    )


class TestGetRequiredInputs:
    def test_text2world_rgb_not_required(self):
        gen = _make_generator("text2world")
        assert gen.get_required_inputs() == {"rgb": False}

    def test_image2world_rgb_required(self):
        gen = _make_generator("image2world")
        assert gen.get_required_inputs() == {"rgb": True}

    def test_video2world_rgb_required(self):
        gen = _make_generator("video2world")
        assert gen.get_required_inputs() == {"rgb": True}


class TestExecute:
    def test_delegates_to_executor(self):
        gen = _make_generator("video2world")
        gen.executor.execute.return_value = (True, "/tmp/out.mp4")

        success, path = gen.execute(
            prompt="test",
            input_media_path="/tmp/input.mp4",
            control_inputs={"edge": "/tmp/edge.mp4"},
            output_path="/tmp/out.mp4",
        )

        gen.executor.execute.assert_called_once_with(
            "test", "/tmp/input.mp4", "/tmp/out.mp4"
        )
        assert success is True
        assert path == "/tmp/out.mp4"

    def test_control_inputs_ignored(self):
        gen = _make_generator("video2world")
        gen.executor.execute.return_value = (True, "/tmp/out.mp4")

        gen.execute("test", "/tmp/in.mp4", {"edge": "e", "depth": "d"}, "/tmp/out.mp4")

        call_args = gen.executor.execute.call_args[0]
        assert len(call_args) == 3  # prompt, input_media_path, output_path

    def test_text2world_none_input(self):
        gen = _make_generator("text2world")
        gen.executor.execute.reset_mock()
        gen.executor.execute.return_value = (True, "/tmp/out.mp4")

        success, _ = gen.execute("generate scene", None, {}, "/tmp/out.mp4")
        gen.executor.execute.assert_called_once_with(
            "generate scene", None, "/tmp/out.mp4"
        )
        assert success is True


class TestSeedProperty:
    def test_seed_reads_from_executor(self):
        gen = _make_generator(seed=123)
        gen.executor.seed = 123
        assert gen.seed == 123

    def test_seed_setter_updates_executor(self):
        gen = _make_generator(seed=42)
        gen.seed = 99
        assert gen.executor.seed == 99
