# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a folder-only PAS augmented dataset from image-edit outputs.

The input is the output of:
1. combine_panes.py, which writes <person_key>.json pane metadata.
2. modules/cli.py image-edit runs, which write
   <person_key>/aug_<n>/output.jpg and output_metadata.json.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

logger = logging.getLogger(__name__)

SKIP_VALUES = {"", "none", "not visible", "unknown", "other"}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def normalize_attribute_key(key: str) -> str:
    return key.replace("_", " ")


def _usable(value: Any) -> bool:
    return str(value).strip().lower() not in SKIP_VALUES


def _garment_phrase(attrs: Dict[str, Any], color_key: str, type_key: str) -> str:
    color = str(attrs.get(color_key, "")).strip()
    garment_type = str(attrs.get(type_key, "")).strip()
    return " ".join(part for part in (color, garment_type) if _usable(part))


def build_queries(attrs: Dict[str, Any]) -> Dict[str, List[str]]:
    top = _garment_phrase(attrs, "top outer color", "top outer type")
    bottom = _garment_phrase(attrs, "bottom color", "bottom type")
    shoes = _garment_phrase(attrs, "shoe color", "shoe type")
    parts = [part for part in (top, bottom, shoes) if part]
    clothing = ", ".join(parts)

    return {
        "easy": parts,
        "medium": [f"person wearing {clothing}"] if clothing else ["person"],
        "hard": [f"Person wearing {clothing}."] if clothing else ["Person."],
    }


def split_pane_image(
    combined_image_path: Path, metadata_path: Path, output_dir: Path
) -> List[Path]:
    metadata = load_json(metadata_path)
    image_order = metadata["image_order"]
    widths = metadata["widths"]
    height = int(metadata["height"])
    original_resolutions = metadata.get("original_resolutions", [])

    if not image_order:
        raise ValueError(f"No image_order entries in {metadata_path}")
    if len(widths) != len(image_order):
        raise ValueError(
            f"metadata widths length ({len(widths)}) must match image_order ({len(image_order)})"
        )

    combined_image = cv2.imread(str(combined_image_path), cv2.IMREAD_COLOR)
    if combined_image is None:
        raise ValueError(f"Invalid image file: {combined_image_path}")

    current_height, current_width = combined_image.shape[:2]
    if current_height != height:
        aspect_ratio = current_width / current_height
        new_width = int(height * aspect_ratio)
        combined_image = cv2.resize(
            combined_image, (new_width, height), interpolation=cv2.INTER_AREA
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    x_offset = 0

    for i, (filename, width) in enumerate(zip(image_order, widths)):
        width = int(width)
        x_end = x_offset + width
        cropped_image = combined_image[0:height, x_offset:x_end]

        if original_resolutions and i < len(original_resolutions):
            original = original_resolutions[i]
            cropped_image = cv2.resize(
                cropped_image,
                (int(original["width"]), int(original["height"])),
                interpolation=cv2.INTER_AREA,
            )

        output_file = output_dir / filename
        if not cv2.imwrite(
            str(output_file), cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        ):
            raise OSError(f"Failed to write split image: {output_file}")

        created_files.append(output_file)
        x_offset = x_end

    return created_files


def build_entry(
    *,
    person_key: str,
    aug_dir: Path,
    pane_metadata_path: Path,
    output_dir: Path,
    dataset_prefix: str,
) -> Optional[Dict[str, Any]]:
    output_image = aug_dir / "output.jpg"
    output_metadata_path = aug_dir / "output_metadata.json"
    if not output_image.exists() or not output_metadata_path.exists():
        logger.warning("Missing output files in %s, skipping", aug_dir)
        return None

    output_metadata = load_json(output_metadata_path)
    selections = output_metadata.get("selections", {})
    if not isinstance(selections, dict):
        logger.warning("Invalid selections in %s, skipping", output_metadata_path)
        return None

    aug_suffix = aug_dir.name.replace("aug_", "", 1)
    augmented_key = f"{person_key}_aug{aug_suffix}"
    split_output_dir = output_dir / "augmented_imgs" / augmented_key
    split_images = split_pane_image(output_image, pane_metadata_path, split_output_dir)
    selected_attributes = {
        normalize_attribute_key(key): value for key, value in selections.items()
    }

    return {
        "person_key": augmented_key,
        "person_id": person_key,
        "source_person_key": person_key,
        "augmentation_id": aug_dir.name,
        "images": [
            f"{dataset_prefix}/augmented_imgs/{augmented_key}/{image_path.name}"
            for image_path in split_images
        ],
        "selected_attributes": selected_attributes,
        "queries": build_queries(selected_attributes),
        "attribute_verification": output_metadata.get("attribute_verification", {}),
    }


def process_augmented_outputs(
    base_dir: Path, augmented_folders: List[Path], output_dir: Path
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    dataset_prefix = output_dir.name

    for augmented_folder in augmented_folders:
        if not augmented_folder.is_dir():
            logger.warning("%s is not a directory, skipping", augmented_folder)
            continue

        person_dirs = sorted(
            d
            for d in augmented_folder.iterdir()
            if d.is_dir() and not d.name.endswith("_configs")
        )
        for person_dir in person_dirs:
            person_key = person_dir.name
            pane_metadata_path = base_dir / f"{person_key}.json"
            if not pane_metadata_path.exists():
                logger.warning("Missing pane metadata for %s, skipping", person_key)
                continue

            aug_dirs = sorted(
                d
                for d in person_dir.iterdir()
                if d.is_dir() and d.name.startswith("aug_")
            )
            for aug_dir in aug_dirs:
                entry = build_entry(
                    person_key=person_key,
                    aug_dir=aug_dir,
                    pane_metadata_path=pane_metadata_path,
                    output_dir=output_dir,
                    dataset_prefix=dataset_prefix,
                )
                if entry is not None:
                    entries.append(entry)

    return entries


def compute_metadata(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_ids = {entry["source_person_key"] for entry in entries}
    total_images = sum(len(entry["images"]) for entry in entries)

    return {
        "description": "Augmented PAS Data from folder-only input",
        "total_ids": len(entries),
        "original_ids": len(source_ids),
        "total_images": total_images,
        "image_structure": "augmented_imgs/{person_key}/{image_filename}",
        "query_format": {
            "levels": ["easy", "medium", "hard"],
            "description": "Queries are built from selected augmentation attributes",
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Create folder-only augmented PAS dataset from image-edit outputs"
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Directory containing <person_key>.json pane metadata from combine_panes.py",
    )
    parser.add_argument(
        "--augmented-folders",
        nargs="+",
        required=True,
        help="One or more directories containing <person_key>/aug_<n>/output.jpg outputs",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for split images and output dataset JSON",
    )
    parser.add_argument(
        "--output-json",
        default="augmented_data.json",
        help="Output JSON filename under --output-dir",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    augmented_folders = [
        Path(folder).expanduser().resolve() for folder in args.augmented_folders
    ]

    entries = process_augmented_outputs(base_dir, augmented_folders, output_dir)
    output_data = {
        "metadata": compute_metadata(entries),
        "entries": entries,
    }

    output_json_path = output_dir / args.output_json
    save_json(output_data, output_json_path)

    logger.info("Dataset creation complete!")
    logger.info("Output directory: %s", output_dir)
    logger.info("Images: %s", output_dir / "augmented_imgs")
    logger.info("Labels: %s", output_json_path)
    logger.info("Total entries: %d", len(entries))
    logger.info("Total images: %d", output_data["metadata"]["total_images"])


if __name__ == "__main__":
    main()
