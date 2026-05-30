# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LocalCosmosPredictExecutor."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock cosmos_oss before importing the executor
mock_init = MagicMock()
with patch.dict("sys.modules", {"cosmos_oss": MagicMock(), "cosmos_oss.init": mock_init}):
    mock_init.init_environment = MagicMock()
    from generation.cosmos import local_predict_executor as _lpe_module
    from generation.cosmos.local_predict_executor import LocalCosmosPredictExecutor


@pytest.fixture
def executor():
    """Create a LocalCosmosPredictExecutor with mocked cosmos_oss."""
    with patch.object(_lpe_module, "init_environment"):
        return LocalCosmosPredictExecutor(
            inference_type="video2world",
            seed=42,
            guidance=3.0,
            num_steps=35,
            num_output_frames=200,
            enable_autoregressive=True,
            chunk_size=93,
            chunk_overlap=2,
            resolution="none",
            logger=MagicMock(),
            inference_name="test_predict",
            num_processes=4,
            master_port=12341,
            inference_script="/opt/cosmos-predict2.5/examples/inference.py",
            model="14B/post-trained",
        )


@pytest.fixture
def text2world_executor():
    """Create a text2world executor."""
    with patch.object(_lpe_module, "init_environment"):
        return LocalCosmosPredictExecutor(
            inference_type="text2world",
            seed=42,
            guidance=3.0,
            num_steps=35,
            num_output_frames=93,
            enable_autoregressive=None,
            chunk_size=None,
            chunk_overlap=None,
            resolution="none",
            logger=MagicMock(),
            inference_name="test_text2world",
        )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    @pytest.mark.parametrize("inference_type", ["text2world", "image2world", "video2world"])
    def test_valid_inference_types(self, inference_type):
        with patch.object(_lpe_module, "init_environment"):
            exe = LocalCosmosPredictExecutor(
                inference_type=inference_type,
                seed=0, guidance=3.0, num_steps=35,
                num_output_frames=None, enable_autoregressive=None,
                chunk_size=None, chunk_overlap=None, resolution="none",
                logger=MagicMock(), inference_name="test",
            )
            assert exe.inference_type == inference_type

    def test_invalid_inference_type_raises(self):
        with patch.object(_lpe_module, "init_environment"):
            with pytest.raises(ValueError, match="Unsupported inference_type"):
                LocalCosmosPredictExecutor(
                    inference_type="invalid",
                    seed=0, guidance=3.0, num_steps=35,
                    num_output_frames=None, enable_autoregressive=None,
                    chunk_size=None, chunk_overlap=None, resolution="none",
                    logger=MagicMock(), inference_name="test",
                )

    @pytest.mark.parametrize("port", [0, -1, 65536, 100000])
    def test_invalid_master_port_raises(self, port):
        with patch.object(_lpe_module, "init_environment"):
            with pytest.raises(ValueError, match="invalid master_port"):
                LocalCosmosPredictExecutor(
                    inference_type="video2world",
                    seed=0, guidance=3.0, num_steps=35,
                    num_output_frames=None, enable_autoregressive=None,
                    chunk_size=None, chunk_overlap=None, resolution="none",
                    logger=MagicMock(), inference_name="test",
                    master_port=port,
                )

    def test_num_processes_clamped(self):
        with patch.object(_lpe_module, "init_environment"):
            exe = LocalCosmosPredictExecutor(
                inference_type="video2world",
                seed=0, guidance=3.0, num_steps=35,
                num_output_frames=None, enable_autoregressive=None,
                chunk_size=None, chunk_overlap=None, resolution="none",
                logger=MagicMock(), inference_name="test",
                num_processes=-1,
            )
            assert exe.num_processes == 1

    def test_cosmos_root_from_script(self):
        with patch.object(_lpe_module, "init_environment"):
            exe = LocalCosmosPredictExecutor(
                inference_type="video2world",
                seed=0, guidance=3.0, num_steps=35,
                num_output_frames=None, enable_autoregressive=None,
                chunk_size=None, chunk_overlap=None, resolution="none",
                logger=MagicMock(), inference_name="test",
                inference_script="/opt/cosmos-predict2.5/examples/inference.py",
            )
            assert exe.cosmos_root == str(
                Path("/opt/cosmos-predict2.5/examples/inference.py").resolve().parent.parent
            )


# ---------------------------------------------------------------------------
# _build_predict_params
# ---------------------------------------------------------------------------


class TestBuildPredictParams:
    def test_video2world_full_params(self, executor):
        params = executor._build_predict_params("/tmp/prompt.txt", "/tmp/input.mp4")
        assert params["inference_type"] == "video2world"
        assert params["name"] == "test_predict"
        assert params["input_path"] == str(Path("/tmp/input.mp4").resolve())
        assert params["num_steps"] == 35
        assert params["seed"] == 42
        assert params["guidance"] == 3.0
        assert params["resolution"] == "none"
        assert params["num_output_frames"] == 200
        assert params["enable_autoregressive"] is True
        assert params["chunk_size"] == 93
        assert params["chunk_overlap"] == 2

    def test_text2world_omits_input_path(self, text2world_executor):
        params = text2world_executor._build_predict_params("/tmp/prompt.txt")
        assert "input_path" not in params
        assert params["inference_type"] == "text2world"

    def test_optional_fields_excluded_when_none(self):
        """When optional params are None, they must not appear in the spec."""
        with patch.object(_lpe_module, "init_environment"):
            exe = LocalCosmosPredictExecutor(
                inference_type="text2world",
                seed=0, guidance=3.0, num_steps=35,
                num_output_frames=None, enable_autoregressive=None,
                chunk_size=None, chunk_overlap=None, resolution=None,
                logger=MagicMock(), inference_name="test",
            )
        params = exe._build_predict_params("/tmp/prompt.txt")
        assert "num_output_frames" not in params
        assert "enable_autoregressive" not in params
        assert "chunk_size" not in params
        assert "chunk_overlap" not in params
        assert "resolution" not in params

    def test_prompt_path_resolved_to_absolute(self, executor):
        params = executor._build_predict_params("relative/prompt.txt", "/tmp/input.mp4")
        assert Path(params["prompt_path"]).is_absolute()

    def test_input_path_resolved_to_absolute(self, executor):
        params = executor._build_predict_params("/tmp/prompt.txt", "relative/input.mp4")
        assert Path(params["input_path"]).is_absolute()


# ---------------------------------------------------------------------------
# _validate_input_type
# ---------------------------------------------------------------------------


class TestValidateInputType:
    def test_text2world_no_input_ok(self, text2world_executor):
        text2world_executor._validate_input_type(None)  # Should not raise

    def test_text2world_with_input_warns(self, text2world_executor):
        # Upstream ignores input_path for text2world but we warn
        text2world_executor._validate_input_type("/tmp/video.mp4")
        text2world_executor.logger.warning.assert_called_once()

    def test_video2world_mp4_ok(self, executor):
        executor._validate_input_type("/tmp/video.mp4")

    @pytest.mark.parametrize("ext", [".jpg", ".png", ".jpeg", ".webp"])
    def test_video2world_image_extensions_ok(self, executor, ext):
        # Upstream accepts both image and video extensions for video2world
        executor._validate_input_type(f"/tmp/input{ext}")

    def test_video2world_no_input_raises(self, executor):
        with pytest.raises(ValueError, match="requires an input file"):
            executor._validate_input_type(None)

    def test_image2world_image_ok(self):
        with patch.object(_lpe_module, "init_environment"):
            exe = LocalCosmosPredictExecutor(
                inference_type="image2world",
                seed=0, guidance=3.0, num_steps=35,
                num_output_frames=None, enable_autoregressive=None,
                chunk_size=None, chunk_overlap=None, resolution="none",
                logger=MagicMock(), inference_name="test",
            )
            exe._validate_input_type("/tmp/input.jpg")

    def test_image2world_mp4_ok(self):
        # Upstream accepts video extensions for image2world too
        with patch.object(_lpe_module, "init_environment"):
            exe = LocalCosmosPredictExecutor(
                inference_type="image2world",
                seed=0, guidance=3.0, num_steps=35,
                num_output_frames=None, enable_autoregressive=None,
                chunk_size=None, chunk_overlap=None, resolution="none",
                logger=MagicMock(), inference_name="test",
            )
            exe._validate_input_type("/tmp/input.mp4")

    def test_image2world_no_input_raises(self):
        with patch.object(_lpe_module, "init_environment"):
            exe = LocalCosmosPredictExecutor(
                inference_type="image2world",
                seed=0, guidance=3.0, num_steps=35,
                num_output_frames=None, enable_autoregressive=None,
                chunk_size=None, chunk_overlap=None, resolution="none",
                logger=MagicMock(), inference_name="test",
            )
            with pytest.raises(ValueError, match="requires an input file"):
                exe._validate_input_type(None)

    @pytest.mark.parametrize("ext", [".txt", ".pdf", ".csv", ".zip"])
    def test_unsupported_extension_raises(self, executor, ext):
        with pytest.raises(ValueError, match="unsupported file extension"):
            executor._validate_input_type(f"/tmp/file{ext}")


# ---------------------------------------------------------------------------
# execute — happy path and error paths
# ---------------------------------------------------------------------------


class TestExecute:
    def _patch_execute_deps(self):
        """Return context managers for mocking execute dependencies."""
        from unittest.mock import patch as _patch
        return (
            _patch.object(subprocess, "run"),
            _patch.object(_lpe_module, "is_remote_path", return_value=False),
            _patch.object(_lpe_module, "msc"),
        )

    def test_execute_success(self, tmp_path):
        # Create a real inference_script file so exists() check passes
        script = tmp_path / "examples" / "inference.py"
        script.parent.mkdir(parents=True)
        script.write_text("# fake script")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_path = str(output_dir / "result.mp4")

        with patch.object(_lpe_module, "init_environment"):
            exe = LocalCosmosPredictExecutor(
                inference_type="video2world", seed=42, guidance=3.0, num_steps=35,
                num_output_frames=200, enable_autoregressive=True,
                chunk_size=93, chunk_overlap=2, resolution="none",
                logger=MagicMock(), inference_name="test_predict",
                num_processes=4, master_port=12341,
                inference_script=str(script),
            )

        def fake_subprocess_run(cmd, **kwargs):
            o_idx = cmd.index("-o")
            temp_dir = cmd[o_idx + 1]
            (Path(temp_dir) / "test_predict.mp4").write_bytes(b"fake")

        with (
            patch.object(subprocess, "run", side_effect=fake_subprocess_run) as mock_run,
            patch.object(_lpe_module, "is_remote_path", return_value=False),
            patch.object(_lpe_module, "msc"),
            patch.object(exe, "_copy_file_msc", return_value=True),
        ):
            success, result_path = exe.execute("test prompt", "/tmp/input.mp4", output_path)

        assert success is True
        assert result_path == output_path
        mock_run.assert_called_once()

    def test_execute_subprocess_error(self, executor):
        with (
            patch.object(subprocess, "run", side_effect=subprocess.CalledProcessError(1, "torchrun")),
            patch.object(_lpe_module, "is_remote_path", return_value=False),
            patch.object(_lpe_module, "msc"),
        ):
            success, result = executor.execute("prompt", "/tmp/input.mp4", "/tmp/out.mp4")
        assert success is False
        assert result is None

    def test_execute_timeout(self, executor):
        with (
            patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("torchrun", 3600)),
            patch.object(_lpe_module, "is_remote_path", return_value=False),
            patch.object(_lpe_module, "msc"),
        ):
            success, result = executor.execute("prompt", "/tmp/input.mp4", "/tmp/out.mp4")
        assert success is False
        assert result is None

    def test_execute_missing_output(self, executor):
        with (
            patch.object(subprocess, "run"),
            patch.object(_lpe_module, "is_remote_path", return_value=False),
            patch.object(_lpe_module, "msc"),
        ):
            success, result = executor.execute("prompt", "/tmp/input.mp4", "/tmp/out.mp4")
        assert success is False
        assert result is None

    def test_execute_copy_failure(self, executor):
        def fake_subprocess_run(cmd, **kwargs):
            o_idx = cmd.index("-o")
            temp_dir = cmd[o_idx + 1]
            (Path(temp_dir) / "test_predict.mp4").write_bytes(b"fake")

        with (
            patch.object(subprocess, "run", side_effect=fake_subprocess_run),
            patch.object(_lpe_module, "is_remote_path", return_value=False),
            patch.object(_lpe_module, "msc"),
            patch.object(executor, "_copy_file_msc", return_value=False),
        ):
            success, result = executor.execute("prompt", "/tmp/input.mp4", "/tmp/out.mp4")
        assert success is False
        assert result is None

    def test_execute_text2world_no_input(self, tmp_path):
        script = tmp_path / "examples" / "inference.py"
        script.parent.mkdir(parents=True)
        script.write_text("# fake script")

        with patch.object(_lpe_module, "init_environment"):
            exe = LocalCosmosPredictExecutor(
                inference_type="text2world", seed=42, guidance=3.0, num_steps=35,
                num_output_frames=93, enable_autoregressive=None,
                chunk_size=None, chunk_overlap=None, resolution="none",
                logger=MagicMock(), inference_name="test_text2world",
                inference_script=str(script),
            )

        captured_spec = {}

        def fake_subprocess_run(cmd, **kwargs):
            o_idx = cmd.index("-o")
            temp_dir = cmd[o_idx + 1]
            # Read the spec while temp dir still exists
            spec_path = cmd[cmd.index("-i") + 1]
            with open(spec_path) as f:
                captured_spec.update(json.load(f))
            (Path(temp_dir) / "test_text2world.mp4").write_bytes(b"fake")

        with (
            patch.object(subprocess, "run", side_effect=fake_subprocess_run),
            patch.object(_lpe_module, "is_remote_path", return_value=False),
            patch.object(_lpe_module, "msc"),
            patch.object(exe, "_copy_file_msc", return_value=True),
        ):
            success, result = exe.execute("generate a scene", None, "/tmp/out.mp4")

        assert success is True
        assert "input_path" not in captured_spec
        assert captured_spec["inference_type"] == "text2world"


# ---------------------------------------------------------------------------
# _run_torchrun_inference
# ---------------------------------------------------------------------------


class TestRunTorchrun:
    def test_command_includes_model(self, executor, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{}")
        with (
            patch.object(subprocess, "run") as mock_run,
            patch.object(Path, "exists", return_value=True),
        ):
            executor._run_torchrun_inference(spec, tmp_path)
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "14B/post-trained"

    def test_command_without_model(self, text2world_executor, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{}")
        with (
            patch.object(subprocess, "run") as mock_run,
            patch.object(Path, "exists", return_value=True),
        ):
            text2world_executor._run_torchrun_inference(spec, tmp_path)
        cmd = mock_run.call_args[0][0]
        assert "--model" not in cmd

    def test_missing_script_raises(self, executor, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{}")
        with pytest.raises(FileNotFoundError, match="Inference script not found"):
            executor._run_torchrun_inference(spec, tmp_path)
