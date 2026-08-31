# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import patch

import pytest
import yaml

from modules.config_distribution_generation.generate_augmentation_configs import (
    _is_image_to_video_config,
    create_seed_image_t2i_config,
    generate_configs,
    generate_output_paths,
    parse_distribution,
    sample_variable_values,
)


LOGGER = logging.getLogger(__name__)


def test_parse_distribution_accepts_native_dict():
    assert parse_distribution({"clear": 0.5, "cloudy": 0.5}, LOGGER) == {
        "clear": 0.5,
        "cloudy": 0.5,
    }


def test_parse_distribution_accepts_dict_literal_string():
    assert parse_distribution("{'clear': 0.5, 'cloudy': 0.5}", LOGGER) == {
        "clear": 0.5,
        "cloudy": 0.5,
    }


def test_parse_distribution_normalizes_probabilities():
    parsed = parse_distribution({"a": 1.0, "b": 1.0}, LOGGER)
    assert parsed == {"a": 0.5, "b": 0.5}


def test_parse_distribution_does_not_call_builtins_eval():
    with patch("builtins.eval") as mock_eval:
        parse_distribution("{'clear': 1.0}", LOGGER)
        mock_eval.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        "dict(clear=1.0)",  # call expression — not a literal
        "len({'clear': 1.0})",  # call expression
        "[('clear', 1.0)]",  # list, not a mapping
        "1.0",  # bare number
        "",  # empty / invalid syntax
        "{1: 1.0}",  # non-string key
        "{'clear': True}",  # bool is not a valid probability type here
        "{'clear': 'high'}",  # non-numeric value
        "{}",  # empty mapping
        "{'clear': -1.0, 'snowy': 2.0}",  # negative probability
        "{'clear': 0.0, 'snowy': 0.0}",  # nothing to sample from
        "{'clear': 1e400}",  # overflows to inf as a plain literal
    ],
)
def test_parse_distribution_rejects_unsafe_or_invalid_strings(payload):
    with pytest.raises(ValueError, match="Invalid distribution"):
        parse_distribution(payload, LOGGER)


@pytest.mark.parametrize(
    "distribution",
    [
        {"clear": float("nan")},
        {"clear": float("inf")},
        {"clear": float("-inf")},
        {"clear": 1.0, "snowy": float("nan")},
        # Native dicts can carry these from PyYAML: YAML 1.1 parses .nan/.inf as floats.
        yaml.safe_load("clear: .nan"),
        yaml.safe_load("clear: .inf"),
        {"clear": 1e308, "snowy": 1e308},  # finite values, total overflows to inf
    ],
)
def test_parse_distribution_rejects_non_finite_probabilities(distribution):
    with pytest.raises(ValueError, match="Invalid distribution"):
        parse_distribution(distribution, LOGGER)


def test_parse_distribution_allows_zero_weight_categories():
    """Zero-probability options are used in real workflow configs (e.g. dawn: 0.0)."""
    assert parse_distribution(
        {"dawn": 0.0, "day": 0.5, "dusk": 0.0, "night": 0.5}, LOGGER
    ) == {"dawn": 0.0, "day": 0.5, "dusk": 0.0, "night": 0.5}


def test_parse_distribution_rejects_code_without_side_effects(tmp_path):
    """Non-literal strings must fail closed with no filesystem side effects.

    Uses a write expression that would create a marker if evaluated with eval().
    """
    marker = tmp_path / "distribution_side_effect_marker"
    # Deliberately not a shell payload: proves calls are not executed.
    payload = f"open({str(marker)!r}, 'w').write('pwned') or {{'a': 1.0}}"

    with pytest.raises(ValueError, match="Invalid distribution"):
        parse_distribution(payload, LOGGER)

    assert not marker.exists()


def test_sample_variable_values_rejects_non_literal_distribution_string():
    variables = {"weather": "dict(clear=1.0)"}
    with pytest.raises(ValueError, match="Invalid distribution"):
        sample_variable_values(variables, {}, n_samples=1, seed=0, logger=LOGGER)


def test_sample_variable_values_accepts_dict_literal_string():
    samples, _ = sample_variable_values(
        {"weather": "{'clear': 1.0}"},
        {},
        n_samples=2,
        seed=0,
        logger=LOGGER,
    )
    assert all(sample["weather"] == "clear" for sample in samples)


def test_conditional_variables_follow_parent_distribution():
    variables = {
        "weather_condition": {"clear": 1.0, "snowy": 0.0},
    }
    conditional_variables = {
        "road_condition": {
            "depends_on": "weather_condition",
            "distributions": {
                "clear": {"dry": 1.0},
                "snowy": {"snowy": 1.0},
            },
        }
    }

    samples, samples_base = sample_variable_values(
        variables, conditional_variables, n_samples=4, seed=42, logger=LOGGER
    )

    assert all(sample["weather_condition"] == "clear" for sample in samples)
    assert all(sample["road_condition"] == "dry" for sample in samples)
    assert all(sample_base["road_condition"] == "dry" for sample_base in samples_base)


def test_conditional_variables_can_chain():
    variables = {
        "weather_condition": {"rainy": 1.0},
    }
    conditional_variables = {
        "road_condition": {
            "depends_on": "weather_condition",
            "distributions": {
                "rainy": {"wet": 1.0},
            },
        },
        "road_visibility": {
            "depends_on": "road_condition",
            "distributions": {
                "wet": {"reflective": 1.0},
            },
        },
    }

    samples, _ = sample_variable_values(
        variables, conditional_variables, n_samples=2, seed=7, logger=LOGGER
    )

    assert all(sample["road_condition"] == "wet" for sample in samples)
    assert all(sample["road_visibility"] == "reflective" for sample in samples)


def test_conditional_variables_require_all_parent_values():
    variables = {
        "weather_condition": {"clear": 0.5, "snowy": 0.5},
    }
    conditional_variables = {
        "road_condition": {
            "depends_on": "weather_condition",
            "distributions": {
                "clear": {"dry": 1.0},
            },
        }
    }

    with pytest.raises(
        ValueError,
        match="missing distributions for weather_condition values: snowy",
    ):
        sample_variable_values(
            variables, conditional_variables, n_samples=1, seed=1, logger=LOGGER
        )


def test_image_to_video_outputs_use_video_key_and_mp4_extension():
    config = {"augmentation": {"model": {"name": "cosmos3-image2video"}}}

    assert _is_image_to_video_config(config, "/data/retail.jpg") is True
    outputs = generate_output_paths(
        "/data/retail.jpg",
        aug_index=2,
        data_dir="/data",
        output_root="/out",
        output_extension=".mp4",
    )

    assert outputs == {
        "video": "/out/retail/aug_2/output.mp4",
        "caption": "/out/retail/aug_2/output_prompt.txt",
        "metadata": "/out/retail/aug_2/output_metadata.json",
    }


def test_seed_image_t2i_config_applies_runtime_values():
    example = {
        "data": [
            {
                "inputs": {"rgb": "/unused.png"},
                "output": {
                    "video": "/template/seed.png",
                    "caption": "/template/seed_prompt.txt",
                    "metadata": "/template/seed_metadata.json",
                },
            }
        ],
        "captioning": {"llm": {"variables": {}}},
        "augmentation": {"parameters": {"seed": 0, "guidance_scale": 1.0}},
    }
    workflow = {
        "env_type": "Retail Store",
        "env_description": "A grocery aisle",
        "output_root": "/out",
        "output_basename": "seed_image",
        "seed_base": 100,
        "variation_instructions": ["Variation A", "Variation B"],
        "runtime_overrides": {"augmentation.parameters.guidance_scale": 6.0},
    }

    config = create_seed_image_t2i_config(example, workflow, 1, LOGGER)

    assert config["data"][0]["inputs"]["rgb"] is None
    assert config["data"][0]["output"]["video"] == (
        "/out/retail_store/aug_1/seed_image.png"
    )
    assert config["captioning"]["llm"]["variables"] == {
        "env_type": ["Retail Store"],
        "env_description": ["A grocery aisle"],
        "variation_instruction": ["Variation B"],
    }
    assert config["augmentation"]["parameters"]["seed"] == 101
    assert config["augmentation"]["parameters"]["guidance_scale"] == 6.0


def test_event_video_distribution_generates_runnable_mp4_output(tmp_path):
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "retail.jpg").write_bytes(b"seed")

    example_path = tmp_path / "example.yaml"
    example_path.write_text(
        yaml.safe_dump(
            {
                "data": [
                    {
                        "inputs": {"rgb": "/template/input.jpg"},
                        "output": {
                            "video": "/template/output.mp4",
                            "caption": "/template/output_prompt.txt",
                            "metadata": "/template/output_metadata.json",
                            "evaluation": "${data.0.output.video}_evaluation.json",
                        },
                    }
                ],
                "captioning": {"llm": {"variables": {}}},
                "augmentation": {
                    "model": {"name": "cosmos3-image2video"},
                    "parameters": {"seed": 42},
                },
            },
            sort_keys=False,
        )
    )

    config_dir = tmp_path / "generated_configs"
    output_dir = tmp_path / "generated_videos"
    generated = generate_configs(
        {
            "workflow_type": "event_video_gen_i2v",
            "data_dir": str(seed_dir),
            "n_augmentations": 1,
            "config_output": str(config_dir),
            "example_augmentation_config": str(example_path),
            "output_root": str(output_dir),
            "variables": {"anomaly_type": {"person_falling": 1.0}},
        },
        LOGGER,
    )

    assert len(generated) == 1
    config = yaml.safe_load((config_dir / "config_retail_aug0.yaml").read_text())
    data = config["data"][0]
    assert data["inputs"]["rgb"] == str(seed_dir / "retail.jpg")
    assert data["output"]["video"] == str(
        output_dir / "retail" / "aug_0" / "output.mp4"
    )
    assert data["output"]["evaluation"] == ("${data.0.output.video}_evaluation.json")
    assert config["captioning"]["llm"]["variables"] == {
        "anomaly_type": ["person_falling"]
    }
