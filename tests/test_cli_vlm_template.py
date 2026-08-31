# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt persistence and provenance tests for VLM-template captioning."""

import json
import logging
import sys

import yaml

from captioning.template_captioner import TemplateCaptioner
from modules import cli


LOGGER = logging.getLogger("test-cli-vlm-template")
TEMPLATE = (
    "{scene_caption}\n\n"
    "Requested event:\n{event_type_text}\n\n"
    "Motion:\n{motion_level_text}\n\n"
    "Aftermath:\n{aftermath_text}\n\n"
    "Preserve the original viewpoint, people, and fixed objects."
)
ATTRIBUTES = {
    "event_type": {
        "person_falling": (
            "The existing foreground worker wearing yellow PPE loses balance "
            "and falls onto the concrete floor."
        )
    },
    "motion_level": {
        "natural": "Use clear, physically plausible motion with realistic timing."
    },
    "aftermath": {
        "remains_visible": "The affected worker remains visible through the end."
    },
}
SELECTIONS = {
    "event_type": "person_falling",
    "motion_level": "natural",
    "aftermath": "remains_visible",
}


class FakeVLM:
    def __init__(self, caption):
        self.caption = caption
        self.calls = []

    def get_caption(self, media_path):
        self.calls.append(media_path)
        return self.caption


class FakeGenerator:
    def __init__(self):
        self.seed = 42
        self.calls = []

    def execute(self, prompt, rgb_path, control_inputs, output_media_path):
        self.calls.append(
            {
                "prompt": prompt,
                "rgb_path": rgb_path,
                "control_inputs": control_inputs,
                "output_media_path": output_media_path,
            }
        )
        return True, output_media_path


class FakeTracker:
    def __init__(self):
        self.completed = None

    def update(self, **kwargs):
        pass

    def complete(self, **kwargs):
        self.completed = kwargs


class FakeAttributeVerifier:
    def __init__(self):
        self.calls = []

    def verify_video_attributes(
        self, video_path, selected_variables, variable_options, extra_questions
    ):
        self.calls.append(
            {
                "video_path": video_path,
                "selected_variables": selected_variables,
                "variable_options": variable_options,
                "extra_questions": extra_questions,
            }
        )
        return True, {"summary": {"total_checks": len(selected_variables)}}


def test_cli_persists_exact_prompt_selections_scene_and_generator_input(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "warehouse.png"
    image_path.write_bytes(b"mocked-vlm-input")
    prompt_path = tmp_path / "resolved_prompt.txt"
    metadata_path = tmp_path / "metadata.json"
    output_path = tmp_path / "generated.mp4"
    scene = (
        "A fixed high-angle view shows a warehouse aisle, storage racks, boxes, "
        "and three existing workers."
    )
    expected_prompt = (
        f"{scene}\n\n"
        f"Requested event:\n{ATTRIBUTES['event_type']['person_falling']}\n\n"
        f"Motion:\n{ATTRIBUTES['motion_level']['natural']}\n\n"
        f"Aftermath:\n{ATTRIBUTES['aftermath']['remains_visible']}\n\n"
        "Preserve the original viewpoint, people, and fixed objects."
    )
    config = {
        "data": [
            {
                "inputs": {
                    "rgb": str(image_path),
                    "prompt_attributes": SELECTIONS,
                },
                "output": {
                    "video": str(output_path),
                    "caption": str(prompt_path),
                    "metadata": str(metadata_path),
                },
            }
        ],
        "endpoints": [
            {
                "id": "vlm",
                "role": "vlm",
                "url": "http://localhost:8000/v1",
                "model": "test/vlm",
            },
            {
                "id": "cosmos3-image2video",
                "role": "image2video",
                "url": "http://localhost:8007",
                "model": "nvidia/Cosmos3-Super-Image2Video",
                "adapter": "openai.video.sync",
            },
            {
                "id": "llm",
                "role": "llm",
                "url": "http://localhost:8002/v1",
                "model": "test/llm",
            },
        ],
        "pipeline": {"retry": 0, "regenerate_caption_on_retry": False},
        "captioning": {
            "vlm": {
                "parser": "instruct",
                "user_prompt": "Describe only visible facts.",
            },
            "template": {"template": TEMPLATE, "attributes": ATTRIBUTES},
        },
        "augmentation": {
            "model": {"name": "cosmos3-image2video", "version": "super"},
            "parameters": {"seed": 42},
        },
        "evaluators": [{"attribute_verification": {"enabled": True}}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    fake_vlm = FakeVLM(scene)
    captioner = TemplateCaptioner(
        template=TEMPLATE,
        attributes=ATTRIBUTES,
        vlm_captioner=fake_vlm,
        logger=LOGGER,
    )
    generator = FakeGenerator()
    attribute_verifier = FakeAttributeVerifier()
    tracker = FakeTracker()
    monkeypatch.setattr(cli, "load_secrets", lambda logger: None)
    monkeypatch.setattr(cli, "create_captioner", lambda *args, **kwargs: captioner)
    monkeypatch.setattr(cli, "create_generator", lambda *args, **kwargs: generator)
    monkeypatch.setattr(
        cli,
        "_init_evaluators",
        lambda *args, **kwargs: (
            None,
            attribute_verifier,
            {"enabled": True},
            None,
        ),
    )
    monkeypatch.setattr(cli, "NVCFProgressTracker", lambda logger=None: tracker)
    monkeypatch.setattr(sys, "argv", ["cli.py", "--config", str(config_path)])
    monkeypatch.setenv("OVERWRITE_CAPTION", "true")

    cli.main()

    assert fake_vlm.calls == [str(image_path)]
    assert prompt_path.read_text() == expected_prompt
    assert generator.calls[0]["prompt"] == expected_prompt
    metadata = json.loads(metadata_path.read_text())
    assert metadata["prompt"] == expected_prompt
    assert metadata["selections"] == SELECTIONS
    assert metadata["prompt_builder"] == "vlm_template"
    assert metadata["scene_description"] == scene
    assert metadata["scene_description_source"] == "vlm"
    assert metadata["attribute_verification"]["passed"] is True
    assert "prompt_intent" not in metadata
    assert "event_template" not in metadata
    assert tracker.completed == {"metadata": {"totalSamplesProcessed": 1}}
    assert attribute_verifier.calls == [
        {
            "video_path": str(output_path),
            "selected_variables": SELECTIONS,
            "variable_options": {
                variable_name: list(catalog)
                for variable_name, catalog in ATTRIBUTES.items()
            },
            "extra_questions": [],
        }
    ]

    sameer_config = {
        "captioning": {
            "llm": {
                "variables": {
                    "anomaly_type": ["person_falling"],
                    "env_type": ["warehouse"],
                }
            }
        }
    }
    assert cli._attribute_verification_inputs(sameer_config) == (
        {"anomaly_type": "person_falling", "env_type": "warehouse"},
        {
            "anomaly_type": ["person_falling"],
            "env_type": ["warehouse"],
        },
    )


def test_cached_prompt_keeps_selections_without_claiming_a_new_build():
    captioner = TemplateCaptioner(
        template=TEMPLATE,
        attributes=ATTRIBUTES,
        vlm_captioner=FakeVLM("A warehouse scene."),
        logger=LOGGER,
    )
    sample = {
        "inputs": {
            "rgb": "/data/warehouse.png",
            "prompt_attributes": SELECTIONS,
        },
        "output": {},
    }
    config = {"captioning": {"template": {}}}
    captioner.get_caption_for_sample(sample, "/data/warehouse.png")
    assert cli._prompt_builder_provenance(captioner)["prompt_builder"] == "vlm_template"
    assert cli._captioning_selections(config, sample) == SELECTIONS

    captioner.reset_provenance()

    assert cli._prompt_builder_provenance(captioner) == {}
    assert cli._captioning_selections(config, sample) == SELECTIONS
    assert cli._attribute_verification_inputs(config, sample) == (
        SELECTIONS,
        {key: [value] for key, value in SELECTIONS.items()},
    )


def test_generate_prompt_retains_legacy_get_caption_fallback():
    class LegacyCaptioner:
        def __init__(self):
            self.calls = []

        def get_caption(self, media_path):
            self.calls.append(media_path)
            return "Existing captioner output."

    captioner = LegacyCaptioner()
    sample = {"output": {"caption": "/unused/prompt.txt"}}

    prompt = cli._generate_prompt(
        sample,
        captioner,
        LOGGER,
        rgb_path="/data/input.mp4",
        write_prompt=False,
    )

    assert prompt == "Existing captioner output."
    assert captioner.calls == ["/data/input.mp4"]
