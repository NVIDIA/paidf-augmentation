# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Cosmos Predict factory wiring."""

import os
from unittest.mock import MagicMock, patch

from generation.factory import _resolve_seed, _PREDICT_MODEL_MAP


# ---------------------------------------------------------------------------
# _PREDICT_MODEL_MAP
# ---------------------------------------------------------------------------


class TestPredictModelMap:
    def test_2b_mapping(self):
        assert _PREDICT_MODEL_MAP["2B"] == "2B/post-trained"

    def test_2b_distilled_mapping(self):
        assert _PREDICT_MODEL_MAP["2B-distilled"] == "2B/distilled"

    def test_14b_mapping(self):
        assert _PREDICT_MODEL_MAP["14B"] == "14B/post-trained"

    def test_unknown_version_passthrough(self):
        unknown = "custom-model"
        result = _PREDICT_MODEL_MAP.get(unknown, unknown)
        assert result == "custom-model"


# ---------------------------------------------------------------------------
# _resolve_seed
# ---------------------------------------------------------------------------


class TestResolveSeed:
    def test_none_generates_seed(self):
        seed = _resolve_seed(None, MagicMock())
        assert isinstance(seed, int)
        assert seed > 0

    def test_string_none(self):
        seed = _resolve_seed("None", MagicMock())
        assert isinstance(seed, int)

    def test_string_none_lowercase(self):
        seed = _resolve_seed("none", MagicMock())
        assert isinstance(seed, int)

    def test_explicit_int(self):
        assert _resolve_seed(42, MagicMock()) == 42

    def test_string_int(self):
        assert _resolve_seed("123", MagicMock()) == 123


# ---------------------------------------------------------------------------
# _create_cosmos_predict — test via the CosmosPredictGenerator constructor args
# ---------------------------------------------------------------------------


class TestCreateCosmosPredict:
    """Test factory wiring by mocking the CosmosPredictGenerator constructor."""

    def _call_factory(self, params=None, local_params=None, endpoints=None, version=None):
        """Call _create_cosmos_predict with a mocked CosmosPredictGenerator."""
        with patch("generation.cosmos_predict_generator.CosmosPredictGenerator.__init__", return_value=None) as mock_init:
            from generation.factory import _create_cosmos_predict
            _create_cosmos_predict(
                params or {}, local_params or {}, endpoints or {},
                "local", version, MagicMock(),
            )
            return mock_init

    def test_basic_creation(self):
        mock_init = self._call_factory(
            params={"inference_type": "video2world", "num_output_frames": 200,
                    "num_steps": 35, "seed": 42, "guidance": 3.0,
                    "inference_name": "test", "resolution": "none"},
        )
        mock_init.assert_called_once()
        kwargs = mock_init.call_args[1]
        assert kwargs["inference_type"] == "video2world"
        assert kwargs["num_output_frames"] == 200
        assert kwargs["seed"] == 42
        assert kwargs["model"] is None

    def test_model_version_14b(self):
        mock_init = self._call_factory(params={"seed": 42}, version="14B")
        kwargs = mock_init.call_args[1]
        assert kwargs["model"] == "14B/post-trained"

    def test_model_version_2b_distilled(self):
        mock_init = self._call_factory(params={"seed": 42}, version="2B-distilled")
        kwargs = mock_init.call_args[1]
        assert kwargs["model"] == "2B/distilled"

    def test_model_version_none(self):
        mock_init = self._call_factory(params={"seed": 42}, version=None)
        kwargs = mock_init.call_args[1]
        assert kwargs["model"] is None

    def test_endpoint_from_env_var(self):
        with patch.dict(os.environ, {"COSMOS_PREDICT_ENDPOINT_URL": "http://env:9999/"}):
            mock_init = self._call_factory(
                params={"seed": 42},
                endpoints={"cosmos_predict": {"url": "http://config:8888/"}},
            )
        kwargs = mock_init.call_args[1]
        assert kwargs["endpoint"] == "http://env:9999/"

    def test_endpoint_from_config(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COSMOS_PREDICT_ENDPOINT_URL", None)
            mock_init = self._call_factory(
                params={"seed": 42},
                endpoints={"cosmos_predict": {"url": "http://config:8888/"}},
            )
        kwargs = mock_init.call_args[1]
        assert kwargs["endpoint"] == "http://config:8888/"

    def test_default_params(self):
        mock_init = self._call_factory()
        kwargs = mock_init.call_args[1]
        assert kwargs["inference_type"] == "video2world"
        assert kwargs["num_steps"] == 35
        assert kwargs["guidance"] == 3.0
        assert kwargs["inference_name"] == "cosmos_predict_inference"
        assert kwargs["resolution"] == "none"

    def test_local_params_passed_through(self):
        mock_init = self._call_factory(
            params={"seed": 42},
            local_params={"num_processes": 8, "master_port": 54321},
        )
        kwargs = mock_init.call_args[1]
        assert kwargs["num_processes"] == 8
        assert kwargs["master_port"] == 54321
