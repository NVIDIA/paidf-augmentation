# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory coverage for VLM-template prompting and existing strategies."""

import logging
from unittest.mock import patch

import pytest

from aug_utils.schema import PipelineConfig
from captioning.factory import create_captioner
from captioning.file_captioner import FileCaptioner
from captioning.llm_captioner import LLMCaptioner
from captioning.template_captioner import TemplateCaptioner
from captioning.text_captioner import TextCaptioner
from captioning.vlm_captioner import VLMCaptioner
from captioning.vlm_llm_captioner import VLMLLMCaptioner


LOGGER = logging.getLogger("test-template-factory")
TEMPLATE = "{scene_caption}\n{event_type_text}\n{motion_level_text}\n{aftermath_text}"
ATTRIBUTES = {
    "event_type": {"person_falling": "A worker falls."},
    "motion_level": {"natural": "Use natural motion."},
    "aftermath": {"remains_visible": "Keep the worker visible."},
}


@pytest.fixture(autouse=True)
def clear_captioning_endpoint_env(monkeypatch):
    for name in (
        "VLM_ENDPOINT_URL",
        "VLM_ENDPOINT_MODEL",
        "LLM_ENDPOINT_URL",
        "LLM_ENDPOINT_MODEL",
        "LLM_CAPTION_ENDPOINT_URL",
        "LLM_CAPTION_ENDPOINT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def _endpoint(role):
    return {
        "id": role,
        "role": role,
        "url": f"http://{role}:8000/v1",
        "model": f"{role}/model",
    }


def _template_config():
    return {
        "data": [
            {
                "inputs": {
                    "rgb": "/tmp/frame.png",
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
            {**_endpoint("vlm"), "id": "unused-vlm"},
            {
                **_endpoint("vlm"),
                "id": "selected-vlm",
                "url": "http://selected:8000/v1",
                "model": "selected/model",
                "api_key_env": "SELECTED_VLM_API_KEY",
            },
        ],
        "captioning": {
            "vlm": {
                "endpoint_id": "selected-vlm",
                "parser": "instruct",
                "user_prompt": "Describe visible facts.",
            },
            "template": {"template": TEMPLATE, "attributes": ATTRIBUTES},
        },
    }


def test_factory_builds_template_captioner_without_llm_and_honors_vlm_selector(
    monkeypatch,
):
    monkeypatch.setenv("SELECTED_VLM_API_KEY", "selected-key")
    config = PipelineConfig(**_template_config())

    with patch("generation.adapters.openai_chat.OpenAI"):
        captioner = create_captioner(config, LOGGER)

    assert isinstance(captioner, TemplateCaptioner)
    assert captioner.vlm_captioner.endpoint == "http://selected:8000/v1"
    assert captioner.vlm_captioner.model == "selected/model"
    assert captioner.vlm_captioner.adapter.api_key == "selected-key"
    assert all(endpoint.role != "llm" for endpoint in config.endpoints)


@pytest.mark.parametrize(
    ("captioning", "match"),
    [
        (
            {"template": {"template": TEMPLATE, "attributes": ATTRIBUTES}},
            "requires captioning.vlm",
        ),
        (
            {
                "vlm": {"user_prompt": "Describe."},
                "llm": {"variables": {"event": ["fall"]}},
                "template": {"template": TEMPLATE, "attributes": ATTRIBUTES},
            },
            "cannot be combined",
        ),
    ],
)
def test_factory_rejects_invalid_template_combinations(captioning, match):
    with pytest.raises(ValueError, match=match):
        create_captioner({"captioning": captioning}, LOGGER)


def test_raw_dict_factory_validates_template_contract():
    config = _template_config()
    config["captioning"]["template"]["template"] = "{scene_caption}"

    with (
        patch("generation.adapters.openai_chat.OpenAI"),
        pytest.raises(ValueError, match="must contain exactly one"),
    ):
        create_captioner(config, LOGGER)


@pytest.mark.parametrize(
    ("captioning", "endpoints", "expected_type"),
    [
        (
            {"vlm": {"user_prompt": "Describe."}},
            [_endpoint("vlm")],
            VLMCaptioner,
        ),
        (
            {"llm": {"variables": {"event": ["fall"]}}},
            [_endpoint("llm")],
            LLMCaptioner,
        ),
        (
            {
                "vlm": {"user_prompt": "Describe."},
                "llm": {"variables": {"event": ["fall"]}},
            },
            [_endpoint("vlm"), _endpoint("llm")],
            VLMLLMCaptioner,
        ),
    ],
)
def test_existing_endpoint_backed_factory_modes_are_unchanged(
    captioning, endpoints, expected_type
):
    config = {"endpoints": endpoints, "captioning": captioning}
    with patch("generation.adapters.openai_chat.OpenAI"):
        captioner = create_captioner(config, LOGGER)
    assert isinstance(captioner, expected_type)


def test_existing_text_and_file_factory_modes_are_unchanged(tmp_path):
    text_captioner = create_captioner(
        {"captioning": {"llm": {"text": "A fixed prompt."}}}, LOGGER
    )
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("A prompt from a file.\n")
    file_captioner = create_captioner(
        {"captioning": {"llm": {"file_path": str(prompt_file)}}}, LOGGER
    )

    assert isinstance(text_captioner, TextCaptioner)
    assert text_captioner.get_caption("unused") == "A fixed prompt."
    assert isinstance(file_captioner, FileCaptioner)
    assert file_captioner.get_caption("unused") == "A prompt from a file."
