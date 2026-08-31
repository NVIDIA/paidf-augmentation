# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a folder-only Image Attribute Augmentation dataset from image-edit outputs.

The input is the output of:
1. combine_panes.py, which writes <person_key>.json pane metadata.
2. modules/cli.py image-edit runs, which write a combined output image.

Both the standard <person_key>/aug_<n>/ layout and Cosmos-style
<person_key>/<n>/ run directories are supported. A single pane metadata JSON and
a single run directory may also be passed directly.
"""

import argparse
import base64
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import multistorageclient as msc
import numpy as np

logger = logging.getLogger(__name__)

SKIP_VALUES = {"", "none", "not visible", "unknown", "other"}
ACCESSORY_COLORS = {
    "beige",
    "black",
    "blue",
    "brown",
    "camouflage",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}
ACCESSORY_COLOR_SYNONYMS = {
    "tan": "beige",
    "khaki": "beige",
    "cream": "beige",
    "navy": "blue",
    "light blue": "blue",
    "dark blue": "blue",
    "olive": "green",
    "lime": "green",
    "teal": "green",
    "gray": "grey",
    "light gray": "grey",
    "dark gray": "grey",
    "light grey": "grey",
    "dark grey": "grey",
    "charcoal": "grey",
    "silver": "grey",
    "maroon": "red",
    "burgundy": "red",
    "crimson": "red",
    "off-white": "white",
}
UNSUPPORTED_ACCESSORY_COLOR_WORDS = {
    "clear",
    "gold",
    "silver",
    "dark",
    "light",
    "striped",
    "transparent",
}
VISUAL_ATTRIBUTE_PROMPT = """Inspect this single pedestrian crop and return JSON only:
{"accessories": ["visible color wearable accessory"], "viewpoint": "front view"}

Accessories must contain only clearly visible wearable or clothing-related accessories.
Each accessory must include its visible color when color can be determined. Accessory
colors must use only this primary color vocabulary: beige, black, blue, brown,
camouflage, green, grey, orange, pink, purple, red, white, yellow.
Map fine-grained colors to the nearest primary color: tan/khaki/cream -> beige,
navy/light blue -> blue, olive/lime/teal -> green, charcoal/silver -> grey,
maroon/burgundy/crimson -> red, off-white -> white. Do not use unsupported color
words such as clear, gold, silver, dark, light, striped, or transparent in the
accessory phrase.
Examples: black hat, black glasses, red scarf, brown belt, grey watch,
yellow bracelet, black backpack, brown handbag, white purse, blue umbrella.
If the accessory is visible but color is not reliable, return the accessory noun
without guessing the color.

Do not put non-clothing objects in accessories. Examples of non-accessories:
mug, phone, box, package, papers, bicycle, shopping cart.

Return an empty accessories list when no wearable/clothing accessory is present,
visible, or certain. Viewpoint must be exactly one of: front view, side view, or
back view. Choose the closest of these three values.
"""


REMOTE_PATH_PREFIXES = ("s3://", "msc://", "gs://", "az://", "https://")
IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def storage_path(path: str) -> msc.Path:
    """Create a MultiStorage path without resolving remote URIs locally."""
    value = str(path)
    if value.startswith(REMOTE_PATH_PREFIXES):
        return msc.Path(value)
    return msc.Path(Path(value).expanduser().resolve())


def load_json(path: msc.Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: msc.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def normalize_attribute_key(key: str) -> str:
    return key.replace("_", " ")


def _usable(value: Any) -> bool:
    return str(value).strip().lower() not in SKIP_VALUES


def normalize_accessory_phrase(value: str) -> str:
    """Normalize optional leading accessory color to the primary Image Attribute Augmentation vocabulary."""
    text = re.sub(r"\s+", " ", value.strip().lower())
    if not _usable(text):
        return ""
    for source, target in sorted(
        ACCESSORY_COLOR_SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        prefix = f"{source} "
        if text.startswith(prefix):
            return f"{target} {text[len(prefix) :]}".strip()
    parts = text.split(" ")
    if not parts:
        return ""
    while parts and parts[0] in UNSUPPORTED_ACCESSORY_COLOR_WORDS:
        parts = parts[1:]
    parts = [part for part in parts if part not in UNSUPPORTED_ACCESSORY_COLOR_WORDS]
    if parts and parts[0] in ACCESSORY_COLORS:
        return " ".join(parts).strip()
    return " ".join(parts).strip()


def normalize_visual_attributes(answer: str) -> Dict[str, Any]:
    """Parse Gemma's per-image accessories and viewpoint answer."""
    match = re.search(r"\{.*\}", answer, flags=re.DOTALL)
    if not match:
        raise ValueError("Gemma answer did not contain a JSON object")
    parsed = json.loads(match.group(0))

    raw_accessories = parsed.get("accessories", [])
    if not isinstance(raw_accessories, list):
        raw_accessories = []
    accessories = [
        normalized
        for value in raw_accessories
        if isinstance(value, str)
        for normalized in [normalize_accessory_phrase(value)]
        if normalized
    ]

    viewpoint = str(parsed.get("viewpoint", "")).strip().lower()
    allowed_viewpoints = {
        "front view",
        "side view",
        "back view",
    }
    if viewpoint not in allowed_viewpoints:
        viewpoint = ""

    return {"accessories": accessories, "viewpoint": viewpoint}


def create_gemma_extractor(
    endpoint: str, model: str, api_key_env: str
) -> Callable[[msc.Path], Dict[str, Any]]:
    """Create a Gemma VLM callable used once for each split image crop."""
    modules_dir = str(Path(__file__).resolve().parents[1])
    if modules_dir not in sys.path:
        sys.path.insert(0, modules_dir)
    from generation.adapters.openai_chat import OpenAIChatAdapter

    adapter = OpenAIChatAdapter.for_chat(
        endpoint,
        model,
        logger,
        role="vlm",
        timeout=120,
        api_key_env=api_key_env,
    )

    def extract(image_path: msc.Path) -> Dict[str, Any]:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract factual person attributes from one image. "
                    "Do not infer details that are not clearly visible."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISUAL_ATTRIBUTE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ]
        last_error = None
        for _attempt in range(2):
            try:
                response = adapter.chat(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=256,
                    stream=False,
                )
            except Exception as exc:
                if "temperature" not in str(exc).lower():
                    raise
                response = adapter.chat(
                    messages=messages,
                    max_tokens=256,
                    stream=False,
                )
            try:
                return normalize_visual_attributes(
                    response.choices[0].message.content or ""
                )
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "Failed to parse VLM visual attributes for %s: %s",
                    image_path,
                    exc,
                )
        logger.warning(
            "Using empty visual attributes for %s after VLM parse failures: %s",
            image_path,
            last_error,
        )
        return {"accessories": [], "viewpoint": ""}

    return extract


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


def resolve_pane_metadata(base_path: msc.Path, person_key: str) -> Optional[msc.Path]:
    """Resolve pane metadata from a direct JSON path or supported directory layout."""
    if base_path.name.lower().endswith(".json"):
        if base_path.stem != person_key:
            logger.warning(
                "Pane metadata %s is being used for person %s", base_path, person_key
            )
        return base_path if base_path.exists() else None

    candidates = [
        base_path / f"{person_key}.json",
        base_path / person_key / f"{person_key}.json",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def is_augmentation_dir_name(name: str) -> bool:
    """Return whether a directory name represents an augmentation run."""
    return name.startswith("aug_") or name.isdigit()


def augmentation_suffix(name: str) -> str:
    """Normalize aug_0 and Cosmos-style 0 directory names to the same suffix."""
    return name[4:] if name.startswith("aug_") else name


def resolve_output_image(aug_dir: msc.Path) -> Optional[msc.Path]:
    """Find the generated pane image, preferring the standard output.jpg name."""
    preferred = aug_dir / "output.jpg"
    if preferred.exists():
        return preferred

    candidates = sorted(
        (
            path
            for path in aug_dir.iterdir()
            if not path.is_dir() and Path(path.name).suffix.lower() in IMAGE_SUFFIXES
        ),
        key=str,
    )
    if len(candidates) == 1:
        logger.info("Using generated image %s", candidates[0])
        return candidates[0]
    if candidates:
        logger.warning(
            "Multiple generated images found in %s; name the intended image output.jpg",
            aug_dir,
        )
    else:
        logger.warning("No generated JPG image found in %s", aug_dir)
    return None


def iter_augmentation_runs(
    augmented_folder: msc.Path, direct_person_key: Optional[str] = None
):
    """Yield (person_key, run_dir) from batch, person, or direct-run inputs."""
    if is_augmentation_dir_name(augmented_folder.name):
        yield direct_person_key or augmented_folder.parent.name, augmented_folder
        return

    children = sorted(
        (path for path in augmented_folder.iterdir() if path.is_dir()), key=str
    )
    run_dirs = [path for path in children if is_augmentation_dir_name(path.name)]
    if run_dirs:
        person_key = direct_person_key or augmented_folder.name
        for run_dir in run_dirs:
            yield person_key, run_dir
        return

    for person_dir in children:
        if person_dir.name.endswith("_configs"):
            continue
        for run_dir in sorted(
            (
                path
                for path in person_dir.iterdir()
                if path.is_dir() and is_augmentation_dir_name(path.name)
            ),
            key=str,
        ):
            yield person_dir.name, run_dir


def split_pane_image(
    combined_image_path: msc.Path,
    metadata_path: msc.Path,
    output_dir: msc.Path,
) -> List[msc.Path]:
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

    image_bytes = combined_image_path.read_bytes()
    combined_image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
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
        encoded, buffer = cv2.imencode(
            ".jpg", cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        if not encoded:
            raise OSError(f"Failed to write split image: {output_file}")
        output_file.write_bytes(buffer.tobytes())

        created_files.append(output_file)
        x_offset = x_end

    return created_files


def build_entries(
    *,
    person_key: str,
    aug_dir: msc.Path,
    pane_metadata_path: msc.Path,
    output_dir: msc.Path,
    dataset_prefix: str,
    visual_attribute_extractor: Optional[Callable[[msc.Path], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    output_image = resolve_output_image(aug_dir)
    output_metadata_path = aug_dir / "output_metadata.json"
    if output_image is None:
        return []

    output_metadata: Dict[str, Any] = {}
    if output_metadata_path.exists():
        output_metadata = load_json(output_metadata_path)
    else:
        logger.warning(
            "Missing %s; splitting image without augmentation attributes",
            output_metadata_path,
        )
    selections = output_metadata.get("selections", {})
    if not isinstance(selections, dict):
        logger.warning(
            "Invalid selections in %s; using no selections", output_metadata_path
        )
        selections = {}

    aug_suffix = augmentation_suffix(aug_dir.name)
    augmented_key = f"{person_key}_aug{aug_suffix}"
    split_output_dir = output_dir / "augmented_imgs" / augmented_key
    split_images = split_pane_image(output_image, pane_metadata_path, split_output_dir)
    selected_attributes = {
        normalize_attribute_key(key): value for key, value in selections.items()
    }

    entries = []
    for image_path in split_images:
        visual_attributes = {"accessories": [], "viewpoint": ""}
        if visual_attribute_extractor is not None:
            visual_attributes = visual_attribute_extractor(image_path)
        relative_image = (
            f"{dataset_prefix}/augmented_imgs/{augmented_key}/{image_path.name}"
        )
        entries.append(
            {
                "image_id": f"{augmented_key}/{image_path.name}",
                "person_key": augmented_key,
                "person_id": person_key,
                "source_person_key": person_key,
                "augmentation_id": aug_dir.name,
                "images": [relative_image],
                "attributes": {**selections, **visual_attributes},
                "selected_attributes": selected_attributes,
                "queries": build_queries(selected_attributes),
                "attribute_verification": output_metadata.get(
                    "attribute_verification", {}
                ),
            }
        )
    return entries


def process_augmented_outputs(
    base_dir: msc.Path,
    augmented_folders: List[msc.Path],
    output_dir: msc.Path,
    visual_attribute_extractor: Optional[Callable[[msc.Path], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    dataset_prefix = output_dir.name
    direct_person_key = (
        base_dir.stem if base_dir.name.lower().endswith(".json") else None
    )

    for augmented_folder in augmented_folders:
        if not augmented_folder.is_dir():
            logger.warning("%s is not a directory, skipping", augmented_folder)
            continue

        for person_key, aug_dir in iter_augmentation_runs(
            augmented_folder, direct_person_key=direct_person_key
        ):
            pane_metadata_path = resolve_pane_metadata(base_dir, person_key)
            if pane_metadata_path is None:
                logger.warning("Missing pane metadata for %s, skipping", person_key)
                continue
            entries.extend(
                build_entries(
                    person_key=person_key,
                    aug_dir=aug_dir,
                    pane_metadata_path=pane_metadata_path,
                    output_dir=output_dir,
                    dataset_prefix=dataset_prefix,
                    visual_attribute_extractor=visual_attribute_extractor,
                )
            )

    return entries


def compute_metadata(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_ids = {entry["source_person_key"] for entry in entries}
    total_images = sum(len(entry["images"]) for entry in entries)

    return {
        "description": "Image Attribute Augmentation data from folder-only input",
        "total_entries": len(entries),
        "total_ids": len({entry["person_key"] for entry in entries}),
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
        description="Create folder-only Image Attribute Augmentation dataset from image-edit outputs"
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help=(
            "Local or MultiStorage pane metadata JSON, or a directory containing "
            "<person_key>.json metadata from combine_panes.py"
        ),
    )
    parser.add_argument(
        "--augmented-folders",
        nargs="+",
        required=True,
        help=(
            "One or more local or MultiStorage batch roots, person directories, "
            "or direct run directories. Supports <person_key>/aug_<n>/ and "
            "Cosmos-style <person_key>/<n>/ layouts"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local or MultiStorage directory for split images and dataset JSON",
    )
    parser.add_argument(
        "--output-json",
        default="augmented_data.json",
        help="Output JSON filename under --output-dir",
    )
    parser.add_argument(
        "--vlm-endpoint",
        help="OpenAI-compatible Gemma VLM base URL used for per-image attributes",
    )
    parser.add_argument(
        "--vlm-model",
        help="Gemma VLM model name used for per-image attributes",
    )
    parser.add_argument(
        "--vlm-api-key-env",
        default="VLM_API_KEY",
        help="Environment variable containing the Gemma API key",
    )
    args = parser.parse_args()

    modules_dir = str(Path(__file__).resolve().parents[1])
    if modules_dir not in sys.path:
        sys.path.insert(0, modules_dir)
    from aug_utils.nvcf import load_secrets

    load_secrets(logger)

    base_dir = storage_path(args.base_dir)
    output_dir = storage_path(args.output_dir)
    augmented_folders = [storage_path(folder) for folder in args.augmented_folders]

    if bool(args.vlm_endpoint) != bool(args.vlm_model):
        parser.error("--vlm-endpoint and --vlm-model must be provided together")
    visual_attribute_extractor = None
    if args.vlm_endpoint:
        visual_attribute_extractor = create_gemma_extractor(
            args.vlm_endpoint, args.vlm_model, args.vlm_api_key_env
        )

    entries = process_augmented_outputs(
        base_dir,
        augmented_folders,
        output_dir,
        visual_attribute_extractor=visual_attribute_extractor,
    )
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
