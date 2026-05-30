# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "modules"
    / "data_processing"
    / "create_PAS_augmented_dataset.py"
)


def _load_create_pas_augmented_dataset(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())

    spec = importlib.util.spec_from_file_location(
        "create_PAS_augmented_dataset", SCRIPT_PATH
    )
    create_pas_augmented_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(create_pas_augmented_dataset)
    return create_pas_augmented_dataset


def test_create_pas_augmented_dataset_builds_queries_from_selected_attributes(monkeypatch):
    create_pas_augmented_dataset = _load_create_pas_augmented_dataset(monkeypatch)
    queries = create_pas_augmented_dataset.build_queries(
        {
            "top outer color": "beige",
            "top outer type": "hoodie",
            "bottom color": "black",
            "bottom type": "jeans",
            "shoe color": "white",
            "shoe type": "sneakers",
        }
    )

    assert queries["easy"] == ["beige hoodie", "black jeans", "white sneakers"]
    assert queries["medium"] == [
        "person wearing beige hoodie, black jeans, white sneakers"
    ]
    assert queries["hard"] == [
        "Person wearing beige hoodie, black jeans, white sneakers."
    ]


def test_create_pas_augmented_dataset_runs_folder_only_command(tmp_path, monkeypatch):
    create_pas_augmented_dataset = _load_create_pas_augmented_dataset(monkeypatch)
    base_dir = tmp_path / "panes"
    aug_dir = tmp_path / "augmented_outputs" / "person_0001" / "aug_0"
    output_dir = tmp_path / "augmented_dataset"
    base_dir.mkdir()
    aug_dir.mkdir(parents=True)
    (base_dir / "person_0001.json").write_text(
        json.dumps(
            {
                "image_order": ["a.jpg", "b.jpg"],
                "widths": [10, 20],
                "height": 512,
                "original_resolutions": [
                    {"width": 10, "height": 20},
                    {"width": 20, "height": 20},
                ],
            }
        )
    )
    (aug_dir / "output.jpg").write_bytes(b"fake image")
    (aug_dir / "output_metadata.json").write_text(
        json.dumps(
            {
                "selections": {
                    "top_outer_color": "beige",
                    "top_outer_type": "hoodie",
                    "bottom_color": "black",
                    "bottom_type": "jeans",
                    "shoe_color": "white",
                    "shoe_type": "sneakers",
                },
                "attribute_verification": {"passed": True},
            }
        )
    )

    def fake_split_pane_image(_combined_image_path, _metadata_path, split_output_dir):
        split_output_dir.mkdir(parents=True)
        created = [split_output_dir / "a.jpg", split_output_dir / "b.jpg"]
        for path in created:
            path.write_bytes(b"fake image")
        return created

    monkeypatch.setattr(
        create_pas_augmented_dataset, "split_pane_image", fake_split_pane_image
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "create_PAS_augmented_dataset.py",
            "--base-dir",
            str(base_dir),
            "--augmented-folders",
            str(tmp_path / "augmented_outputs"),
            "--output-dir",
            str(output_dir),
            "--output-json",
            "augmented_data.json",
        ],
    )

    create_pas_augmented_dataset.main()

    data = json.loads((output_dir / "augmented_data.json").read_text())
    entry = data["entries"][0]
    assert entry["person_key"] == "person_0001_aug0"
    assert entry["person_id"] == "person_0001"
    assert entry["selected_attributes"]["top outer color"] == "beige"
    assert entry["queries"]["easy"] == ["beige hoodie", "black jeans", "white sneakers"]
    assert entry["images"] == [
        "augmented_dataset/augmented_imgs/person_0001_aug0/a.jpg",
        "augmented_dataset/augmented_imgs/person_0001_aug0/b.jpg",
    ]
