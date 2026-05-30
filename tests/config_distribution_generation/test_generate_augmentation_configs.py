# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from modules.config_distribution_generation.generate_augmentation_configs import (
    sample_variable_values,
)


LOGGER = logging.getLogger(__name__)


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
