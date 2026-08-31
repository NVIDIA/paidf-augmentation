# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified pipeline configuration schema."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aug_utils.schema import (
    ModelNameEnum,
    PipelineConfig,
)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _ep(id, role, url, model="test", adapter=None) -> dict:
    """Build an endpoint-registry list entry."""
    entry = {"id": id, "role": role, "url": url, "model": model}
    if adapter is not None:
        entry["adapter"] = adapter
    return entry


def _minimal_config(**overrides) -> dict:
    """Return a minimal valid config dict for cosmos-transfer2.5."""
    base = {
        "data": [
            {
                "inputs": {"rgb": "/tmp/test.mp4"},
                "output": {
                    "video": "/tmp/out.mp4",
                    "caption": "/tmp/cap.txt",
                    "metadata": "/tmp/meta.json",
                },
            }
        ],
        "endpoints": [
            _ep(
                "cosmos_transfer",
                "video_transfer",
                "http://localhost:30002/",
                adapter="nim",
            ),
        ],
        "augmentation": {
            "model": {"name": "cosmos-transfer2.5"},
        },
    }
    base.update(overrides)
    return base


def _minimal_config_with_captioning() -> dict:
    """Return a minimal config with VLM captioning."""
    config = _minimal_config()
    config["endpoints"].append(
        _ep("vlm", "vlm", "http://localhost:9001/v1", "test-vlm")
    )
    config["captioning"] = {
        "vlm": {"system_prompt": "test", "user_prompt": "test"},
    }
    return config


def _minimal_config_image_edit() -> dict:
    """Return a minimal valid config for image-edit."""
    return {
        "data": [
            {
                "inputs": {"rgb": "/tmp/test.png"},
                "output": {
                    "video": "/tmp/out.png",
                    "caption": "/tmp/cap.txt",
                    "metadata": "/tmp/meta.json",
                },
            }
        ],
        "endpoints": [
            _ep("image_edit", "image_edit", "http://localhost:8080/v1", "test-edit"),
        ],
        "augmentation": {
            "model": {"name": "image-edit"},
            "parameters": {"num_inference_steps": 50, "guidance_scale": 1.0},
        },
    }


def _minimal_config_predict() -> dict:
    """Return a minimal valid config for cosmos-predict."""
    return {
        "data": [
            {
                "inputs": {"rgb": "/tmp/test.mp4"},
                "output": {
                    "video": "/tmp/out.mp4",
                    "caption": "/tmp/cap.txt",
                    "metadata": "/tmp/meta.json",
                },
            }
        ],
        "endpoints": [
            _ep(
                "cosmos_predict",
                "video_predict",
                "http://localhost:30003/",
                "predict",
                adapter="nim",
            ),
        ],
        "augmentation": {
            "model": {"name": "cosmos-predict"},
            "parameters": {
                "inference_type": "video2world",
                "num_output_frames": 121,
                "num_steps": 35,
                "enable_autoregressive": True,
                "chunk_size": 49,
                "chunk_overlap": 17,
                "seed": 42,
                "guidance": 3.0,
            },
        },
    }


def _minimal_config_predict_text2world() -> dict:
    """Return a minimal valid config for cosmos-predict text2world (no rgb input)."""
    return {
        "data": [
            {
                "inputs": {},
                "output": {
                    "video": "/tmp/out.mp4",
                    "caption": "/tmp/cap.txt",
                    "metadata": "/tmp/meta.json",
                },
            }
        ],
        "endpoints": [
            _ep(
                "cosmos_predict",
                "video_predict",
                "http://localhost:30003/",
                "predict",
                adapter="nim",
            ),
        ],
        "augmentation": {
            "model": {"name": "cosmos-predict"},
            "parameters": {
                "inference_type": "text2world",
                "num_output_frames": 93,
                "num_steps": 35,
                "seed": 42,
                "guidance": 3.0,
            },
        },
    }


# ---------------------------------------------------------------------------
# Positive tests — valid configs load successfully
# ---------------------------------------------------------------------------


class TestValidConfigs:
    """Configs that should pass validation."""

    def test_minimal_cosmos_transfer(self):
        pc = PipelineConfig(**_minimal_config())
        assert pc.augmentation.model.name == ModelNameEnum.COSMOS_TRANSFER

    def test_minimal_image_edit(self):
        pc = PipelineConfig(**_minimal_config_image_edit())
        assert pc.augmentation.model.name == ModelNameEnum.IMAGE_EDIT

    def test_minimal_cosmos_predict(self):
        pc = PipelineConfig(**_minimal_config_predict())
        assert pc.augmentation.model.name == ModelNameEnum.COSMOS_PREDICT
        assert pc.augmentation.parameters.inference_type.value == "video2world"
        assert pc.augmentation.parameters.num_output_frames == 121
        assert pc.augmentation.parameters.enable_autoregressive is True
        assert pc.augmentation.parameters.chunk_size == 49

    def test_cosmos_predict_text2world(self):
        pc = PipelineConfig(**_minimal_config_predict_text2world())
        assert pc.augmentation.model.name == ModelNameEnum.COSMOS_PREDICT
        assert pc.augmentation.parameters.inference_type.value == "text2world"
        assert pc.data[0].inputs.rgb is None

    def test_cosmos_predict_image2world(self):
        config = _minimal_config_predict()
        config["augmentation"]["parameters"]["inference_type"] = "image2world"
        pc = PipelineConfig(**config)
        assert pc.augmentation.parameters.inference_type.value == "image2world"

    def test_cosmos_predict_invalid_inference_type(self):
        config = _minimal_config_predict()
        config["augmentation"]["parameters"]["inference_type"] = "invalid"
        with pytest.raises(ValidationError):
            PipelineConfig(**config)

    def test_cosmos_predict_with_resolution(self):
        config = _minimal_config_predict()
        config["augmentation"]["parameters"]["resolution"] = "none"
        pc = PipelineConfig(**config)
        assert pc.augmentation.parameters.resolution == "none"

    def test_with_vlm_captioning(self):
        pc = PipelineConfig(**_minimal_config_with_captioning())
        assert pc.captioning.vlm is not None
        assert pc.captioning.llm is None

    def test_with_vlm_llm_captioning(self):
        config = _minimal_config_with_captioning()
        config["endpoints"].append(_ep("llm", "llm", "http://localhost:8001/v1", "llm"))
        config["captioning"]["llm"] = {
            "system_prompt": "test",
            "variables": {"weather": ["rain"]},
        }
        pc = PipelineConfig(**config)
        assert pc.captioning.vlm is not None
        assert pc.captioning.llm is not None
        assert pc.captioning.llm.variables == {"weather": ["rain"]}

    def test_with_evaluators(self):
        config = _minimal_config()
        # attribute_verification needs an 'llm' (question generation) and a 'vlm'
        # (its inline verifier) transport to resolve.
        config["endpoints"].append(_ep("llm", "llm", "http://localhost:9002/v1", "llm"))
        config["endpoints"].append(_ep("vlm", "vlm", "http://localhost:9001/v1", "vlm"))
        config["pipeline"] = {"retry": 1}
        config["evaluators"] = [
            {"hallucination_check": {"enabled": True, "threshold": 0.7}},
            {
                "attribute_verification": {
                    "enabled": True,
                    "vlm_verification": {"system_prompt": "test"},
                }
            },
        ]
        pc = PipelineConfig(**config)
        assert len(pc.evaluators) == 2
        assert pc.pipeline.retry == 1
        assert pc.evaluators[1].attribute_verification.enabled is True
        assert (
            pc.evaluators[1].attribute_verification.vlm_verification.system_prompt
            == "test"
        )

    def test_standalone_vlm_verification(self):
        config = _minimal_config()
        config["endpoints"].append(_ep("vlm", "vlm", "http://localhost:9001/v1", "vlm"))
        config["evaluators"] = [
            {"vlm_verification": {"system_prompt": "standalone test"}},
        ]
        pc = PipelineConfig(**config)
        assert len(pc.evaluators) == 1
        assert pc.evaluators[0].vlm_verification.system_prompt == "standalone test"

    def test_attribute_verification_without_inline_vlm(self):
        config = _minimal_config()
        # An enabled attribute_verification needs both an 'llm' (question
        # generation) and a 'vlm' (the default verifier) transport.
        config["endpoints"].append(_ep("llm", "llm", "http://localhost:9002/v1", "llm"))
        config["endpoints"].append(_ep("vlm", "vlm", "http://localhost:9001/v1", "vlm"))
        config["evaluators"] = [
            {"attribute_verification": {"enabled": True}},
        ]
        pc = PipelineConfig(**config)
        assert pc.evaluators[0].attribute_verification.vlm_verification is None

    def test_with_modalities(self):
        config = _minimal_config()
        config["augmentation"]["modalities"] = {
            "edge": 0.4,
            "depth": 0.4,
            "seg": 0.3,
            "positive_prompt": "cinematic",
            "negative_prompt": "cartoon",
        }
        pc = PipelineConfig(**config)
        assert pc.augmentation.modalities.edge == 0.4
        assert pc.augmentation.modalities.seg == 0.3
        assert pc.augmentation.modalities.positive_prompt == "cinematic"

    def test_with_pipeline_evaluation(self):
        config = _minimal_config()
        config["pipeline"] = {
            "retry": 5,
            "evaluation": {"strict": False, "retain_failures": False},
            "logging": {"enabled": True, "level": "DEBUG"},
        }
        pc = PipelineConfig(**config)
        assert pc.pipeline.retry == 5
        assert pc.pipeline.evaluation.strict is False
        assert pc.pipeline.logging.level == "DEBUG"

    @pytest.mark.parametrize(
        "config_file",
        sorted(CONFIGS_DIR.rglob("config_*.yaml")),
        ids=lambda p: str(p.relative_to(CONFIGS_DIR)),
    )
    def test_load_all_config_files(self, config_file):
        """Every config_*.yaml under configs/ (incl. cookbook/) must validate."""
        config = _load_yaml(str(config_file))
        PipelineConfig(**config)


# ---------------------------------------------------------------------------
# Default value tests
# ---------------------------------------------------------------------------


class TestDefaults:
    """Verify that Pydantic defaults match the tuned values."""

    def test_default_pipeline_settings(self):
        pc = PipelineConfig(**_minimal_config())
        assert pc.pipeline.retry == 1
        assert pc.pipeline.regenerate_caption_on_retry is True
        assert pc.pipeline.logging.enabled is True
        assert pc.pipeline.logging.level == "INFO"
        assert pc.pipeline.evaluation.strict is True
        assert pc.pipeline.evaluation.retain_failures is True

    def test_default_inference_parameters(self):
        pc = PipelineConfig(**_minimal_config_with_captioning())
        vlm_params = pc.captioning.vlm.parameters
        assert vlm_params.temperature == 0.3
        assert vlm_params.top_p == 0.95
        assert vlm_params.frequency_penalty == 1.05
        assert vlm_params.max_tokens == 4096
        assert vlm_params.stream is False
        assert vlm_params.fps == 4.0
        assert vlm_params.max_pixels == 307200

    def test_default_augmentation_parameters(self):
        pc = PipelineConfig(**_minimal_config())
        params = pc.augmentation.parameters
        assert params.sigma == 90
        assert params.seed is None
        assert params.guidance == 3.0
        assert params.num_steps == 35
        # Default ' ' (single space) so Image-Edit-Plus runs with CFG on
        # by default. Set to null in YAML to disable.
        assert params.negative_prompt == " "

    def test_image_edit_negative_prompt_can_be_null(self):
        config = _minimal_config_image_edit()
        config["augmentation"]["parameters"]["negative_prompt"] = None
        pc = PipelineConfig(**config)
        assert pc.augmentation.parameters.negative_prompt is None

    def test_image_edit_negative_prompt_custom_value(self):
        config = _minimal_config_image_edit()
        config["augmentation"]["parameters"]["negative_prompt"] = "blurry, low quality"
        pc = PipelineConfig(**config)
        assert pc.augmentation.parameters.negative_prompt == "blurry, low quality"

    def test_default_hallucination_params(self):
        config = _minimal_config()
        config["evaluators"] = [{"hallucination_check": {}}]
        pc = PipelineConfig(**config)
        hc = pc.evaluators[0].hallucination_check
        assert hc.enabled is True
        assert hc.threshold == 0.682
        assert hc.params.grad_thresh == 10.0


# ---------------------------------------------------------------------------
# Cross-section validation tests
# ---------------------------------------------------------------------------


class TestCrossSectionValidation:
    """Endpoint requirements are enforced across sections."""

    def test_cosmos_transfer_requires_endpoint(self):
        """Every model now needs a resolvable endpoint (no local fallback)."""
        config = _minimal_config()
        config["endpoints"] = []
        with pytest.raises(ValidationError, match="requires a matching"):
            PipelineConfig(**config)

    def test_cosmos_predict_requires_endpoint(self):
        config = _minimal_config_predict()
        config["endpoints"] = []
        with pytest.raises(ValidationError, match="requires a matching"):
            PipelineConfig(**config)

    def test_image_edit_requires_endpoint(self):
        config = _minimal_config_image_edit()
        config["endpoints"] = []
        with pytest.raises(ValidationError, match="requires a matching"):
            PipelineConfig(**config)

    def test_unknown_adapter_rejected(self):
        config = _minimal_config_image_edit()
        config["endpoints"][0]["adapter"] = "not-a-real-adapter"
        with pytest.raises(ValidationError, match="unknown adapter"):
            PipelineConfig(**config)

    def test_vlm_captioning_requires_vlm_endpoint(self, monkeypatch):
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config_with_captioning()
        config["endpoints"] = [ep for ep in config["endpoints"] if ep["role"] != "vlm"]
        with pytest.raises(ValidationError, match="role 'vlm'"):
            PipelineConfig(**config)

    def test_llm_captioning_requires_llm_endpoint(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("LLM_CAPTION_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("vlm", "vlm", "http://x", "x"))
        config["captioning"] = {
            "vlm": {"system_prompt": "t", "user_prompt": "t"},
            "llm": {"system_prompt": "t", "variables": {"a": ["b"]}},
        }
        with pytest.raises(ValidationError, match="role 'llm'"):
            PipelineConfig(**config)

    def test_text_captioning_does_not_require_llm_endpoint(self):
        """'text' field set should work without endpoints.llm."""
        config = _minimal_config()
        config["captioning"] = {
            "llm": {"text": "a fixed prompt"},
        }
        pc = PipelineConfig(**config)
        assert pc.captioning.llm.text == "a fixed prompt"

    def test_file_captioning_does_not_require_llm_endpoint(self):
        """'file_path' field set should work without endpoints.llm."""
        config = _minimal_config()
        config["captioning"] = {
            "llm": {"file_path": "/path/to/prompts.txt"},
        }
        pc = PipelineConfig(**config)
        assert pc.captioning.llm.file_path == "/path/to/prompts.txt"

    def test_text_and_file_path_mutually_exclusive(self):
        config = _minimal_config()
        config["captioning"] = {
            "llm": {"text": "a prompt", "file_path": "/path/to/file.txt"},
        }
        with pytest.raises(ValidationError, match="mutually exclusive"):
            PipelineConfig(**config)

    def test_attribute_verification_requires_llm_endpoint(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()  # only a video_transfer endpoint, no llm/vlm
        config["evaluators"] = [{"attribute_verification": {"enabled": True}}]
        with pytest.raises(ValidationError, match="role 'llm'"):
            PipelineConfig(**config)

    def test_attribute_verification_requires_vlm_endpoint(self, monkeypatch):
        # An enabled attribute_verification needs a VLM too, even with no inline
        # vlm_verification block (it silently no-ops otherwise).
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("llm", "llm", "http://x", "x"))
        config["evaluators"] = [{"attribute_verification": {"enabled": True}}]
        with pytest.raises(ValidationError, match="role 'vlm'"):
            PipelineConfig(**config)

    def test_disabled_attribute_verification_skips_endpoint_check(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()  # no llm/vlm endpoint
        config["evaluators"] = [{"attribute_verification": {"enabled": False}}]
        pc = PipelineConfig(**config)  # disabled -> no transport required
        assert pc.evaluators[0].attribute_verification.enabled is False

    def test_standalone_vlm_verification_requires_vlm_endpoint(self, monkeypatch):
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()  # no vlm endpoint
        config["evaluators"] = [{"vlm_verification": {"system_prompt": "t"}}]
        with pytest.raises(ValidationError, match="role 'vlm'"):
            PipelineConfig(**config)

    def test_evaluator_endpoint_satisfied_by_env(self, monkeypatch):
        """A role env override (no endpoints entry) satisfies the presence check."""
        monkeypatch.setenv("VLM_ENDPOINT_URL", "http://localhost:9001/v1")
        config = _minimal_config()  # no vlm endpoint, but env supplies it
        config["evaluators"] = [{"vlm_verification": {"system_prompt": "t"}}]
        pc = PipelineConfig(**config)
        assert pc.evaluators[0].vlm_verification.system_prompt == "t"

    def test_two_same_role_endpoints_require_id_even_with_env(self, monkeypatch):
        # An env URL override does NOT disambiguate 2+ same-role endpoints (it
        # can't say which one), and runtime select_endpoint() raises on the
        # ambiguous match — so this must fail validation, not crash at startup.
        monkeypatch.setenv("VLM_ENDPOINT_URL", "http://env/v1")
        config = _minimal_config()
        config["endpoints"].append(_ep("vlm-a", "vlm", "http://a/v1", "a"))
        config["endpoints"].append(_ep("vlm-b", "vlm", "http://b/v1", "b"))
        config["evaluators"] = [{"vlm_verification": {"system_prompt": "t"}}]
        with pytest.raises(ValidationError, match="set endpoint_id"):
            PipelineConfig(**config)

    def test_two_same_role_endpoints_ok_when_id_pins_one(self, monkeypatch):
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("vlm-a", "vlm", "http://a/v1", "a"))
        config["endpoints"].append(_ep("vlm-b", "vlm", "http://b/v1", "b"))
        config["evaluators"] = [
            {"vlm_verification": {"system_prompt": "t", "endpoint_id": "vlm-b"}}
        ]
        pc = PipelineConfig(**config)
        assert pc.evaluators[0].vlm_verification.endpoint_id == "vlm-b"


class TestEndpointSelection:
    """BYOM endpoint_id selection guards at PipelineConfig construction."""

    def test_captioning_llm_endpoint_id_not_found(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("LLM_CAPTION_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("llm-a", "llm", "http://x", "x"))
        config["captioning"] = {
            "llm": {
                "system_prompt": "t",
                "variables": {"a": ["b"]},
                "endpoint_id": "does-not-exist",
            }
        }
        with pytest.raises(ValidationError, match="not found"):
            PipelineConfig(**config)

    def test_endpoint_id_wrong_role_rejected(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("LLM_CAPTION_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("vlm-a", "vlm", "http://x", "x"))
        config["endpoints"].append(_ep("llm-a", "llm", "http://x", "x"))
        config["captioning"] = {
            "llm": {
                "system_prompt": "t",
                "variables": {"a": ["b"]},
                "endpoint_id": "vlm-a",
            }
        }
        with pytest.raises(ValidationError, match="has role 'vlm'"):
            PipelineConfig(**config)

    def test_two_llm_endpoints_distinct_selectors_ok(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("LLM_CAPTION_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("llm-cap", "llm", "http://a", "a"))
        config["endpoints"].append(_ep("llm-qg", "llm", "http://b", "b"))
        config["endpoints"].append(_ep("vlm", "vlm", "http://c", "c"))
        config["captioning"] = {
            "llm": {
                "system_prompt": "t",
                "variables": {"a": ["b"]},
                "endpoint_id": "llm-cap",
            }
        }
        config["evaluators"] = [
            {
                "attribute_verification": {
                    "question_generation": {"endpoint_id": "llm-qg"},
                }
            }
        ]
        pc = PipelineConfig(**config)
        assert pc.captioning.llm.endpoint_id == "llm-cap"
        qg = pc.evaluators[0].attribute_verification.question_generation
        assert qg.endpoint_id == "llm-qg"

    def test_ambiguous_vlm_role_without_endpoint_id_rejected(self, monkeypatch):
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        config["endpoints"].append(_ep("vlm-a", "vlm", "http://a", "a"))
        config["endpoints"].append(_ep("vlm-b", "vlm", "http://b", "b"))
        config["captioning"] = {
            "vlm": {"system_prompt": "t", "user_prompt": "t"},
        }
        with pytest.raises(ValidationError, match="endpoint_id"):
            PipelineConfig(**config)

    def test_duplicate_ids_rejected(self):
        config = _minimal_config()
        config["endpoints"].append(_ep("dup", "vlm", "http://a", "a"))
        config["endpoints"].append(_ep("dup", "llm", "http://b", "b"))
        with pytest.raises(ValidationError, match="duplicate endpoint ids"):
            PipelineConfig(**config)

    def test_id_omitted_resolves_by_role(self, monkeypatch):
        monkeypatch.delenv("VLM_ENDPOINT_URL", raising=False)
        config = _minimal_config()
        # An endpoint with no id at all (now optional) plus captioning.vlm.
        config["endpoints"].append(
            {"role": "vlm", "url": "http://localhost:9001/v1", "model": "test-vlm"}
        )
        config["captioning"] = {
            "vlm": {"system_prompt": "t", "user_prompt": "t"},
        }
        pc = PipelineConfig(**config)
        vlm_eps = [ep for ep in pc.endpoints if ep.role == "vlm"]
        assert len(vlm_eps) == 1
        assert vlm_eps[0].id is None


# ---------------------------------------------------------------------------
# Negative tests — invalid configs are rejected
# ---------------------------------------------------------------------------


class TestInvalidConfigs:
    """Configs that must be rejected with clear errors."""

    def test_unknown_model_without_endpoint_rejected(self):
        """BYOM model names are free-form, but must resolve to an endpoint."""
        config = _minimal_config()
        config["augmentation"]["model"]["name"] = "my-unmapped-model"
        with pytest.raises(ValidationError, match="requires a matching"):
            PipelineConfig(**config)

    def test_byom_model_name_with_matching_endpoint_ok(self):
        """A free-form model name resolves by endpoint id."""
        config = _minimal_config()
        config["augmentation"]["model"]["name"] = "my-byom-model"
        config["endpoints"] = [
            _ep("my-byom-model", "image_edit", "http://x", adapter="nim"),
        ]
        # rgb input is supplied by _minimal_config, role image_edit -> rgb ok.
        pc = PipelineConfig(**config)
        assert pc.augmentation.model.name == "my-byom-model"

    def test_missing_data_section(self):
        with pytest.raises(ValidationError):
            PipelineConfig(endpoints=[])

    def test_empty_data_list(self):
        config = _minimal_config()
        config["data"] = []
        with pytest.raises(ValidationError):
            PipelineConfig(**config)

    def test_missing_rgb_rejected_for_non_text2world(self):
        """rgb is required for models that need input media."""
        config = _minimal_config()
        del config["data"][0]["inputs"]["rgb"]
        with pytest.raises(ValidationError, match=r"inputs\.rgb is required"):
            PipelineConfig(**config)

    def test_missing_rgb_allowed_for_text2world(self):
        """text2world allows null/missing rgb since no media is needed."""
        pc = PipelineConfig(**_minimal_config_predict_text2world())
        rgb = pc.data[0].inputs.rgb if pc.data[0].inputs else None
        assert rgb is None

    def test_missing_video_in_output(self):
        config = _minimal_config()
        del config["data"][0]["output"]["video"]
        with pytest.raises(ValidationError):
            PipelineConfig(**config)

    def test_invalid_logging_level(self):
        config = _minimal_config()
        config["pipeline"] = {"logging": {"level": "TRACE"}}
        with pytest.raises(ValidationError):
            PipelineConfig(**config)

    def test_evaluator_retries_are_rejected(self):
        config = _minimal_config()
        config["evaluators"] = [{"hallucination_check": {"retries": 1}}]
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PipelineConfig(**config)

    def test_evaluator_entry_with_zero_types(self):
        config = _minimal_config()
        config["evaluators"] = [{}]
        with pytest.raises(ValidationError, match="exactly one"):
            PipelineConfig(**config)

    def test_evaluator_entry_with_two_types(self):
        config = _minimal_config()
        config["evaluators"] = [
            {
                "hallucination_check": {"enabled": True},
                "vlm_verification": {"system_prompt": "t"},
            }
        ]
        with pytest.raises(ValidationError, match="exactly one"):
            PipelineConfig(**config)

    def test_modality_weight_out_of_range(self):
        config = _minimal_config()
        config["augmentation"]["modalities"] = {"edge": 1.5}
        with pytest.raises(ValidationError):
            PipelineConfig(**config)


# ---------------------------------------------------------------------------
# Extra question tests
# ---------------------------------------------------------------------------


class TestExtraQuestions:
    """Verify extra_questions validation on evaluators."""

    def test_valid_extra_question(self):
        config = _minimal_config()
        config["endpoints"].append(_ep("llm", "llm", "http://localhost:9002/v1", "llm"))
        config["endpoints"].append(_ep("vlm", "vlm", "http://localhost:9001/v1", "vlm"))
        config["evaluators"] = [
            {
                "attribute_verification": {
                    "extra_questions": [
                        {
                            "question": "Is it consistent?",
                            "options": {"A": "Yes", "B": "No"},
                            "correct_answer": "A",
                        }
                    ]
                }
            }
        ]
        pc = PipelineConfig(**config)
        eq = pc.evaluators[0].attribute_verification.extra_questions[0]
        assert eq.question == "Is it consistent?"
        assert eq.correct_answer == "A"

    def test_extra_question_with_reasoning(self):
        config = _minimal_config()
        config["endpoints"].append(_ep("llm", "llm", "http://localhost:9002/v1", "llm"))
        config["endpoints"].append(_ep("vlm", "vlm", "http://localhost:9001/v1", "vlm"))
        config["evaluators"] = [
            {
                "attribute_verification": {
                    "extra_questions": [
                        {
                            "variable": "color",
                            "question": "What color?",
                            "options": {"A": "Red", "B": "Blue"},
                            "correct_answer": "B",
                            "request_reasoning": True,
                        }
                    ]
                }
            }
        ]
        pc = PipelineConfig(**config)
        eq = pc.evaluators[0].attribute_verification.extra_questions[0]
        assert eq.variable == "color"
        assert eq.request_reasoning is True
