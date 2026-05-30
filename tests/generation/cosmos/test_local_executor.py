# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LocalCosmosExecutor's pre-decode format gate wiring.

The format classifier/probe themselves live in ``generation.utils`` and are
tested in ``tests/generation/test_utils.py``; here we cover that the executor
fails fast (and never launches torchrun) on an undecodable input.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from generation.utils import UnsupportedVideoFormatError

# Mock cosmos_oss before importing the executor (matches sibling executor tests).
mock_init = MagicMock()
with patch.dict(
    "sys.modules", {"cosmos_oss": MagicMock(), "cosmos_oss.init": mock_init}
):
    mock_init.init_environment = MagicMock()
    from generation.cosmos import local_executor as _le_module
    from generation.cosmos.local_executor import LocalCosmosExecutor


@pytest.fixture
def executor():
    """Create a LocalCosmosExecutor with mocked cosmos_oss init."""
    with patch.object(_le_module, "init_environment"):
        return LocalCosmosExecutor(
            sigma=90,
            seed=42,
            guidance=3,
            num_steps=35,
            modalities=["edge"],
            weights={"edge": 1.0},
            positive_prompt="",
            negative_prompt="",
            logger=MagicMock(),
            inference_name="test_transfer",
            num_processes=4,
            master_port=12341,
        )


# ---------------------------------------------------------------------------
# _check_input_decodable
# ---------------------------------------------------------------------------


class TestCheckInputDecodable:
    def test_unsupported_raises(self, executor):
        fmt = {
            "codec_name": "h264",
            "profile": "High 4:4:4 Predictive",
            "pix_fmt": "nv12",
        }
        with patch.object(_le_module, "probe_video_format", return_value=fmt):
            with pytest.raises(UnsupportedVideoFormatError, match="4:4:4"):
                executor._check_input_decodable("/data/444.mp4")

    def test_supported_does_not_raise(self, executor):
        fmt = {"codec_name": "h264", "profile": "High", "pix_fmt": "nv12"}
        with patch.object(_le_module, "probe_video_format", return_value=fmt):
            executor._check_input_decodable("/data/420.mp4")  # no raise

    def test_unprobeable_does_not_raise(self, executor):
        # ffprobe unavailable / errored -> empty dict -> do not block.
        with patch.object(_le_module, "probe_video_format", return_value={}):
            executor._check_input_decodable("/data/mystery.mp4")  # no raise
        executor.logger.debug.assert_called()


# ---------------------------------------------------------------------------
# execute() wiring -- gate runs before torchrun
# ---------------------------------------------------------------------------


class TestExecuteFormatGate:
    def test_unsupported_input_fails_before_torchrun(self, executor):
        fmt = {
            "codec_name": "h264",
            "profile": "High 4:4:4 Predictive",
            "pix_fmt": "nv12",
        }
        with (
            patch.object(_le_module, "is_remote_path", return_value=False),
            patch.object(_le_module, "msc"),
            patch.object(_le_module, "probe_video_format", return_value=fmt),
            patch.object(subprocess, "run") as mock_run,
        ):
            success, result = executor.execute(
                "a prompt", "/data/444.mp4", {}, "/data/out.mp4"
            )
        assert success is False
        assert result is None
        # The whole point: we never launch the torchrun subprocess.
        mock_run.assert_not_called()
        # ...and we logged a clear, actionable error.
        assert executor.logger.error.called
        msg = " ".join(str(c.args[0]) for c in executor.logger.error.call_args_list)
        assert "Cannot decode" in msg and "yuv420p" in msg

    def test_supported_input_proceeds_to_torchrun(self, tmp_path):
        # Real inference_script so the _run_torchrun_inference exists() check passes.
        script = tmp_path / "examples" / "inference.py"
        script.parent.mkdir(parents=True)
        script.write_text("# fake script")

        with patch.object(_le_module, "init_environment"):
            exe = LocalCosmosExecutor(
                sigma=90,
                seed=42,
                guidance=3,
                num_steps=35,
                modalities=["edge"],
                weights={"edge": 1.0},
                positive_prompt="",
                negative_prompt="",
                logger=MagicMock(),
                inference_name="test_transfer",
                num_processes=4,
                master_port=12341,
                inference_script=str(script),
            )

        def fake_subprocess_run(cmd, **kwargs):
            o_idx = cmd.index("-o")
            out_dir = cmd[o_idx + 1]
            (Path(out_dir) / "test_transfer.mp4").write_bytes(b"fake")

        fmt = {"codec_name": "h264", "profile": "Main", "pix_fmt": "nv12"}
        with (
            patch.object(_le_module, "is_remote_path", return_value=False),
            patch.object(_le_module, "msc"),
            patch.object(_le_module, "probe_video_format", return_value=fmt),
            patch.object(
                subprocess, "run", side_effect=fake_subprocess_run
            ) as mock_run,
            patch.object(exe, "_copy_file_msc", return_value=True),
        ):
            success, result = exe.execute(
                "a prompt", "/data/420.mp4", {}, "/data/out.mp4"
            )

        assert success is True
        assert result == "/data/out.mp4"
        mock_run.assert_called_once()
