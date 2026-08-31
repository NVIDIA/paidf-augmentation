# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import importlib.util
import sys
from io import BytesIO, StringIO
from types import SimpleNamespace
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "modules"
    / "data_processing"
    / "create_attribute_augmented_dataset.py"
)


def _load_create_attribute_augmented_dataset(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())

    spec = importlib.util.spec_from_file_location(
        "create_attribute_augmented_dataset", SCRIPT_PATH
    )
    create_attribute_augmented_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(create_attribute_augmented_dataset)
    return create_attribute_augmented_dataset


def test_create_attribute_augmented_dataset_builds_queries_from_selected_attributes(
    monkeypatch,
):
    create_attribute_augmented_dataset = _load_create_attribute_augmented_dataset(monkeypatch)
    queries = create_attribute_augmented_dataset.build_queries(
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


def test_normalize_visual_attributes_keeps_empty_accessories(monkeypatch):
    module = _load_create_attribute_augmented_dataset(monkeypatch)

    assert module.normalize_visual_attributes(
        '```json\n{"accessories": ["not visible"], "viewpoint": "Front View"}\n```'
    ) == {"accessories": [], "viewpoint": "front view"}
    assert module.normalize_visual_attributes(
        '```json\n{"accessories": ["not visible"], "viewpoint": "three-quarter view"}\n```'
    ) == {"accessories": [], "viewpoint": ""}
    assert module.normalize_visual_attributes(
        '{"accessories": ["black backpack"], "viewpoint": "side view"}'
    ) == {
        "accessories": ["black backpack"],
        "viewpoint": "side view",
    }
    assert module.normalize_visual_attributes(
        '{"accessories": ["clear glasses", "silver watch", "navy backpack", "white striped hat"], "viewpoint": "back view"}'
    ) == {
        "accessories": ["glasses", "grey watch", "blue backpack", "white hat"],
        "viewpoint": "back view",
    }


def test_storage_path_preserves_multistorage_uri(monkeypatch):
    module = _load_create_attribute_augmented_dataset(monkeypatch)
    seen = []
    monkeypatch.setattr(
        module.msc, "Path", lambda value: seen.append(str(value)) or value
    )

    result = module.storage_path("msc://attribute-data/outputs")

    assert result == "msc://attribute-data/outputs"
    assert seen == ["msc://attribute-data/outputs"]


def test_iter_augmentation_runs_sorts_non_orderable_storage_paths(monkeypatch):
    module = _load_create_attribute_augmented_dataset(monkeypatch)

    class MemoryPath:
        def __init__(self, value, children=()):
            self.value = value
            self._children = children

        @property
        def name(self):
            return self.value.rsplit("/", 1)[-1]

        @property
        def parent(self):
            return MemoryPath(self.value.rsplit("/", 1)[0])

        def __str__(self):
            return self.value

        def is_dir(self):
            return True

        def iterdir(self):
            return iter(self._children)

    aug_1 = MemoryPath("s3://bucket/person_1/aug_1")
    aug_0 = MemoryPath("s3://bucket/person_1/aug_0")
    person = MemoryPath("s3://bucket/person_1", children=(aug_1, aug_0))

    runs = list(module.iter_augmentation_runs(person))

    assert [(key, str(path)) for key, path in runs] == [
        ("person_1", "s3://bucket/person_1/aug_0"),
        ("person_1", "s3://bucket/person_1/aug_1"),
    ]


def test_split_pane_image_uses_storage_bytes(monkeypatch):
    module = _load_create_attribute_augmented_dataset(monkeypatch)
    files = {
        "msc://attribute/metadata/person.json": json.dumps(
            {"image_order": ["a.jpg"], "widths": [3], "height": 2}
        ).encode(),
        "msc://attribute/outputs/output.jpg": b"remote jpeg",
    }

    class MemoryPath:
        def __init__(self, value):
            self.value = value

        def __truediv__(self, child):
            return MemoryPath(f"{self.value.rstrip('/')}/{child}")

        def __str__(self):
            return self.value

        def open(self, mode):
            if "b" in mode:
                stream = BytesIO(files.get(self.value, b""))
            else:
                stream = StringIO(files.get(self.value, b"").decode())
            return stream

        def read_bytes(self):
            return files[self.value]

        def write_bytes(self, data):
            files[self.value] = data

        def mkdir(self, **_kwargs):
            return None

    module.cv2.IMREAD_COLOR = 1
    module.cv2.IMWRITE_JPEG_QUALITY = 2
    module.cv2.imdecode = lambda _buffer, _flag: np.zeros((2, 3, 3), dtype=np.uint8)
    module.cv2.imencode = lambda _ext, _image, _params: (
        True,
        np.frombuffer(b"encoded jpeg", dtype=np.uint8),
    )

    created = module.split_pane_image(
        MemoryPath("msc://attribute/outputs/output.jpg"),
        MemoryPath("msc://attribute/metadata/person.json"),
        MemoryPath("msc://attribute/dataset/person_aug0"),
    )

    assert [str(path) for path in created] == ["msc://attribute/dataset/person_aug0/a.jpg"]
    assert files["msc://attribute/dataset/person_aug0/a.jpg"] == b"encoded jpeg"


def test_create_attribute_augmented_dataset_runs_folder_only_command(tmp_path, monkeypatch):
    create_attribute_augmented_dataset = _load_create_attribute_augmented_dataset(monkeypatch)
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

    def fake_visual_attribute_extractor(image_path):
        if image_path.name == "a.jpg":
            return {"accessories": ["black backpack"], "viewpoint": "back view"}
        return {"accessories": [], "viewpoint": "front view"}

    monkeypatch.setattr(
        create_attribute_augmented_dataset, "split_pane_image", fake_split_pane_image
    )
    monkeypatch.setattr(
        create_attribute_augmented_dataset,
        "create_gemma_extractor",
        lambda *_args: fake_visual_attribute_extractor,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "create_attribute_augmented_dataset.py",
            "--base-dir",
            str(base_dir),
            "--augmented-folders",
            str(tmp_path / "augmented_outputs"),
            "--output-dir",
            str(output_dir),
            "--output-json",
            "augmented_data.json",
            "--vlm-endpoint",
            "https://gemma.example/v1",
            "--vlm-model",
            "google/gemma-4-31B-it",
        ],
    )

    create_attribute_augmented_dataset.main()

    data = json.loads((output_dir / "augmented_data.json").read_text())
    assert len(data["entries"]) == 2
    first, second = data["entries"]
    assert first["person_key"] == "person_0001_aug0"
    assert first["person_id"] == "person_0001"
    assert first["image_id"] == "person_0001_aug0/a.jpg"
    assert first["attributes"]["top_outer_color"] == "beige"
    assert first["attributes"]["accessories"] == ["black backpack"]
    assert first["attributes"]["viewpoint"] == "back view"
    assert first["selected_attributes"]["top outer color"] == "beige"
    assert first["queries"]["easy"] == ["beige hoodie", "black jeans", "white sneakers"]
    assert first["images"] == [
        "augmented_dataset/augmented_imgs/person_0001_aug0/a.jpg"
    ]
    assert second["image_id"] == "person_0001_aug0/b.jpg"
    assert second["attributes"]["top_outer_color"] == "beige"
    assert second["attributes"]["accessories"] == []
    assert second["attributes"]["viewpoint"] == "front view"
    assert second["images"] == [
        "augmented_dataset/augmented_imgs/person_0001_aug0/b.jpg"
    ]
    assert data["metadata"]["total_entries"] == 2
    assert data["metadata"]["total_ids"] == 1


def test_direct_json_and_cosmos_numeric_run_without_metadata(tmp_path, monkeypatch):
    module = _load_create_attribute_augmented_dataset(monkeypatch)
    person_key = "00000_ICFG_PEDES"
    pane_metadata = tmp_path / "preprocessing" / person_key / f"{person_key}.json"
    run_dir = tmp_path / "cosmos" / person_key / "0"
    output_dir = tmp_path / "result"
    pane_metadata.parent.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    pane_metadata.write_text(
        json.dumps({"image_order": ["a.jpg"], "widths": [10], "height": 512})
    )
    (run_dir / "generated.jpg").write_bytes(b"fake image")

    def fake_split(_image_path, _metadata_path, split_output_dir):
        split_output_dir.mkdir(parents=True)
        output = split_output_dir / "a.jpg"
        output.write_bytes(b"split image")
        return [output]

    monkeypatch.setattr(module, "split_pane_image", fake_split)

    entries = module.process_augmented_outputs(pane_metadata, [run_dir], output_dir)

    assert len(entries) == 1
    assert entries[0]["person_key"] == f"{person_key}_aug0"
    assert entries[0]["person_id"] == person_key
    assert entries[0]["augmentation_id"] == "0"
    assert entries[0]["selected_attributes"] == {}
    assert entries[0]["attribute_verification"] == {}


def test_nested_metadata_and_numeric_runs_under_batch_root(tmp_path, monkeypatch):
    module = _load_create_attribute_augmented_dataset(monkeypatch)
    person_key = "00000_ICFG_PEDES"
    base_dir = tmp_path / "preprocessing"
    pane_metadata = base_dir / person_key / f"{person_key}.json"
    run_dir = tmp_path / "cosmos" / person_key / "1"
    output_dir = tmp_path / "result"
    pane_metadata.parent.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    pane_metadata.write_text(
        json.dumps({"image_order": ["a.jpg"], "widths": [10], "height": 512})
    )
    (run_dir / "output.jpg").write_bytes(b"fake image")
    (run_dir / "output_metadata.json").write_text(
        json.dumps({"selections": {"top_outer_color": "blue"}})
    )

    monkeypatch.setattr(
        module,
        "split_pane_image",
        lambda _image, _metadata, _output: [tmp_path / "a.jpg"],
    )

    entries = module.process_augmented_outputs(
        base_dir, [tmp_path / "cosmos"], output_dir
    )

    assert len(entries) == 1
    assert entries[0]["person_key"] == f"{person_key}_aug1"
    assert entries[0]["selected_attributes"] == {"top outer color": "blue"}
