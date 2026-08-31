# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for deterministic three-attribute prompt construction."""

import logging

import pytest

from captioning.template_captioner import TemplateCaptioner


LOGGER = logging.getLogger("test-template-captioner")
TEMPLATE = (
    "{scene_caption}\n\n"
    "Requested event:\n{event_type_text}\n\n"
    "Motion:\n{motion_level_text}\n\n"
    "Aftermath:\n{aftermath_text}"
)
ATTRIBUTES = {
    "event_type": {
        "person_falling": (
            "The existing foreground PPE worker loses balance and falls "
            "onto the concrete floor."
        )
    },
    "motion_level": {"natural": "Use clear {literal}, physically plausible motion."},
    "aftermath": {
        "remains_visible": "The affected entity remains visible through the end."
    },
}
SELECTIONS = {
    "event_type": "person_falling",
    "motion_level": "natural",
    "aftermath": "remains_visible",
}


class FakeVLM:
    def __init__(self, caption="A fixed warehouse aisle with three visible workers."):
        self.caption = caption
        self.calls = []

    def get_caption(self, media_path):
        self.calls.append(media_path)
        return self.caption


def _make_captioner(*, vlm=None):
    return TemplateCaptioner(
        template=TEMPLATE,
        attributes=ATTRIBUTES,
        vlm_captioner=vlm or FakeVLM(),
        logger=LOGGER,
    )


def _sample(media_path, selections=SELECTIONS):
    inputs = {"rgb": media_path}
    if selections is not None:
        inputs["prompt_attributes"] = selections
    return {"inputs": inputs, "output": {"video": "/tmp/out.mp4"}}


def test_renders_exact_prompt_from_scene_and_three_selected_values():
    vlm = FakeVLM("A high, fixed view of a warehouse aisle.")
    captioner = _make_captioner(vlm=vlm)

    prompt = captioner.get_caption_for_sample(
        _sample("/data/warehouse.png"), "/data/warehouse.png"
    )

    expected = (
        f"{vlm.caption}\n\n"
        f"Requested event:\n{ATTRIBUTES['event_type']['person_falling']}\n\n"
        f"Motion:\n{ATTRIBUTES['motion_level']['natural']}\n\n"
        f"Aftermath:\n{ATTRIBUTES['aftermath']['remains_visible']}"
    )
    assert prompt == expected
    assert "{literal}" in prompt
    assert vlm.calls == ["/data/warehouse.png"]
    assert captioner.last_prompt_builder == "vlm_template"


@pytest.mark.parametrize(
    ("axis", "value_id", "text"),
    [
        (
            "event_type",
            "forklift_collision",
            "The existing forklift collides with the nearest visible storage rack.",
        ),
        ("motion_level", "pronounced", "Make the motion pronounced."),
        ("aftermath", "scene_stabilizes", "The scene stabilizes after the event."),
    ],
)
def test_each_axis_can_vary_independently(axis, value_id, text):
    attributes = {name: dict(catalog) for name, catalog in ATTRIBUTES.items()}
    attributes[axis][value_id] = text
    selections = {**SELECTIONS, axis: value_id}
    vlm = FakeVLM()
    captioner = TemplateCaptioner(
        template=TEMPLATE,
        attributes=attributes,
        vlm_captioner=vlm,
        logger=LOGGER,
    )

    prompt = captioner.get_caption_for_sample(
        _sample("/data/warehouse.png", selections), "/data/warehouse.png"
    )

    assert text in prompt
    assert captioner.last_scene_description == vlm.caption
    assert captioner.last_scene_description_source == "vlm"


@pytest.mark.parametrize("media_path", ["/data/frame.png", "/data/clip.mp4"])
def test_image_and_video_paths_reach_the_vlm(media_path):
    vlm = FakeVLM()
    captioner = _make_captioner(vlm=vlm)

    prompt = captioner.get_caption_for_sample(_sample(media_path), media_path)

    assert vlm.calls == [media_path]
    assert vlm.caption in prompt


@pytest.mark.parametrize(
    "caption",
    ["", '{"scene": "warehouse"}', '```json\n{"scene": "warehouse"}\n```'],
)
def test_invalid_vlm_scene_descriptions_are_rejected(caption):
    captioner = _make_captioner(vlm=FakeVLM(caption))

    with pytest.raises(ValueError, match="VLM"):
        captioner.get_caption_for_sample(_sample("/data/frame.png"), "/data/frame.png")


@pytest.mark.parametrize(
    "selections",
    [
        {"event_type": "person_falling", "motion_level": "natural"},
        {**SELECTIONS, "aftermath": "unknown"},
    ],
)
def test_missing_or_unknown_sample_selection_fails_clearly(selections):
    captioner = _make_captioner()

    with pytest.raises(ValueError, match="must select a defined"):
        captioner.get_caption_for_sample(
            _sample("/data/frame.png", selections), "/data/frame.png"
        )


def test_prompt_attributes_are_required():
    captioner = _make_captioner()

    with pytest.raises(ValueError, match=r"requires data\.inputs\.prompt_attributes"):
        captioner.get_caption_for_sample(
            _sample("/data/frame.png", None), "/data/frame.png"
        )


def test_source_media_is_required():
    captioner = _make_captioner()
    sample = {"inputs": {"prompt_attributes": SELECTIONS}, "output": {}}

    with pytest.raises(ValueError, match=r"data\.inputs\.rgb"):
        captioner.get_caption_for_sample(sample, None)


def test_standard_captioner_call_rejects_missing_sample_selections():
    captioner = _make_captioner()

    with pytest.raises(ValueError, match="get_caption_for_sample"):
        captioner.get_caption("/data/frame.png")
