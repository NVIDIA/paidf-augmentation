# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "modules"
    / "data_processing"
    / "combine_panes.py"
)


def _fake_cv2():
    cv2 = SimpleNamespace(
        IMREAD_COLOR=1,
        IMWRITE_JPEG_QUALITY=1,
        INTER_AREA=1,
        written_shapes={},
    )

    def imread(path, _flags=None):
        name = Path(path).name
        if name == "a.jpg":
            return np.zeros((10, 20, 3), dtype=np.uint8)
        if name == "b.jpg":
            return np.zeros((20, 10, 3), dtype=np.uint8)
        return None

    def resize(image, size, interpolation=None):
        width, height = size
        return np.zeros((height, width, image.shape[2]), dtype=image.dtype)

    def imwrite(path, image, params=None):
        cv2.written_shapes[str(path)] = image.shape
        Path(path).write_bytes(b"fake image")
        return True

    cv2.imread = imread
    cv2.resize = resize
    cv2.imwrite = imwrite
    return cv2


def _load_combine_panes(monkeypatch):
    cv2 = _fake_cv2()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    spec = importlib.util.spec_from_file_location("combine_panes", SCRIPT_PATH)
    combine_panes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(combine_panes)
    return combine_panes, cv2


def test_process_dataset_writes_combined_pane_and_metadata(tmp_path, monkeypatch):
    combine_panes, cv2 = _load_combine_panes(monkeypatch)
    input_dir = tmp_path / "input"
    person_dir = input_dir / "person_0001"
    output_dir = tmp_path / "panes"
    person_dir.mkdir(parents=True)

    (person_dir / "a.jpg").write_bytes(b"fake image")
    (person_dir / "b.jpg").write_bytes(b"fake image")

    combine_panes.process_dataset(input_dir, output_dir)

    metadata = json.loads((output_dir / "person_0001.json").read_text())

    assert cv2.written_shapes[str(output_dir / "person_0001.jpg")] == (512, 1280, 3)
    assert metadata["image_order"] == ["a.jpg", "b.jpg"]
    assert metadata["widths"] == [1024, 256]
    assert metadata["total_width"] == 1280
