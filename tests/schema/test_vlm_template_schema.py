# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema tests for the three-attribute VLM-template strategy."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from aug_utils.schema import PipelineConfig


TEMPLATE = "{scene_caption}\n{event_type_text}\n{motion_level_text}\n{aftermath_text}"


def _config():
    return {
        "data": [
            {
                "inputs": {
                    "rgb": "/tmp/warehouse.png",
                    "prompt_attributes": {
                        "event_type": "person_falling",
                        "motion_level": "natural",
                        "aftermath": "remains_visible",
                    },
                },
                "output": {
                    "video": "/tmp/out.mp4",
                    "caption": "/tmp/prompt.txt",
                    "metadata": "/tmp/metadata.json",
                },
            }
        ],
        "endpoints": [
            {
                "id": "vlm",
                "role": "vlm",
                "url": "http://localhost:8000/v1",
                "model": "test/vlm",
            }
        ],
        "captioning": {
            "vlm": {
                "parser": "instruct",
                "user_prompt": "Describe only visible facts.",
            },
            "template": {
                "template": TEMPLATE,
                "attributes": {
                    "event_type": {
                        "person_falling": "The existing foreground worker falls."
                    },
                    "motion_level": {
                        "natural": "Use clear, physically plausible motion."
                    },
                    "aftermath": {
                        "remains_visible": "The affected entity remains visible."
                    },
                },
            },
        },
    }


def test_schema_accepts_three_selected_ids_and_flat_catalogs():
    config = PipelineConfig(**_config())

    selections = config.data[0].inputs.prompt_attributes
    assert selections.model_dump() == {
        "event_type": "person_falling",
        "motion_level": "natural",
        "aftermath": "remains_visible",
    }
    assert config.captioning.template.attributes.event_type == {
        "person_falling": "The existing foreground worker falls."
    }
    assert config.captioning.llm is None


@pytest.mark.parametrize("change", ["remove_vlm", "add_llm"])
def test_schema_rejects_invalid_template_strategy(change):
    config = _config()
    if change == "remove_vlm":
        del config["captioning"]["vlm"]
        match = "requires captioning.vlm"
    else:
        config["captioning"]["llm"] = {
            "system_prompt": "Write a prompt.",
            "variables": {"event_type": ["person_falling"]},
        }
        match = "cannot be combined"

    with pytest.raises(ValidationError, match=match):
        PipelineConfig(**config)


@pytest.mark.parametrize("attribute_name", ["event_type", "motion_level", "aftermath"])
def test_schema_requires_all_three_sample_selections(attribute_name):
    config = _config()
    del config["data"][0]["inputs"]["prompt_attributes"][attribute_name]

    with pytest.raises(ValidationError, match=attribute_name):
        PipelineConfig(**config)


def test_schema_rejects_unknown_sample_selection():
    config = _config()
    config["data"][0]["inputs"]["prompt_attributes"]["actor"] = "foreground_worker"

    with pytest.raises(ValidationError, match="actor"):
        PipelineConfig(**config)


@pytest.mark.parametrize("attribute_name", ["event_type", "motion_level", "aftermath"])
def test_schema_rejects_selection_not_in_its_catalog(attribute_name):
    config = _config()
    config["data"][0]["inputs"]["prompt_attributes"][attribute_name] = "unknown"

    with pytest.raises(ValidationError, match=rf"{attribute_name}=.*not defined"):
        PipelineConfig(**config)


def test_schema_requires_prompt_attributes_for_template_mode():
    config = _config()
    del config["data"][0]["inputs"]["prompt_attributes"]

    with pytest.raises(ValidationError, match="prompt_attributes is required"):
        PipelineConfig(**config)


def test_schema_rejects_prompt_attributes_without_template_mode():
    config = _config()
    del config["captioning"]["template"]

    with pytest.raises(ValidationError, match="prompt_attributes requires"):
        PipelineConfig(**config)


@pytest.mark.parametrize("attribute_name", ["event_type", "motion_level", "aftermath"])
def test_schema_requires_all_three_non_empty_catalogs(attribute_name):
    config = _config()
    config["captioning"]["template"]["attributes"][attribute_name] = {}

    with pytest.raises(ValidationError, match=attribute_name):
        PipelineConfig(**config)


def test_schema_rejects_unknown_catalog_axis():
    config = _config()
    config["captioning"]["template"]["attributes"]["actor"] = {"worker": "The worker."}

    with pytest.raises(ValidationError, match="actor"):
        PipelineConfig(**config)


def test_schema_rejects_empty_catalog_text():
    config = _config()
    config["captioning"]["template"]["attributes"]["event_type"]["person_falling"] = (
        "   "
    )

    with pytest.raises(ValidationError, match="must not be empty"):
        PipelineConfig(**config)


@pytest.mark.parametrize(
    ("template", "match"),
    [
        ("{scene_caption} {event_type_text} {motion_level_text}", "aftermath_text"),
        (TEMPLATE + " {actor_text}", "actor_text"),
        ("   ", "must not be empty"),
    ],
)
def test_schema_validates_template_placeholders(template, match):
    config = _config()
    config["captioning"]["template"]["template"] = template

    with pytest.raises(ValidationError, match=match):
        PipelineConfig(**config)


def test_existing_vlm_only_config_does_not_require_prompt_attributes():
    config = deepcopy(_config())
    del config["captioning"]["template"]
    del config["data"][0]["inputs"]["prompt_attributes"]

    parsed = PipelineConfig(**config)

    assert parsed.captioning.vlm is not None
    assert parsed.captioning.template is None
