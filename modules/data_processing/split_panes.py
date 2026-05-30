# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Split combined multi-pane images back into individual images.
Uses metadata JSON files created by the pane-combining script.
"""

import json
import logging
from pathlib import Path
from PIL import Image
import argparse

logger = logging.getLogger(__name__)


def split_pane(combined_image_path, metadata_path, output_dir, create_subdirs=True):
    """
    Split a combined multi-pane image back into individual images.
    Resizes each image back to its original resolution.

    Args:
        combined_image_path: Path to the combined image
        metadata_path: Path to the metadata JSON file
        output_dir: Directory to save the split images
        create_subdirs: If True, create subdirectories named after the combined image

    Returns:
        List of paths to the created images
    """
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Extract metadata
    image_order = metadata["image_order"]
    widths = metadata["widths"]
    height = metadata["height"]
    original_resolutions = metadata.get("original_resolutions", [])

    if len(image_order) != 4 or len(widths) != 4:
        raise ValueError(f"Expected 4 images in metadata, got {len(image_order)}")

    # Load combined image
    combined_image = Image.open(combined_image_path)

    # Resize to 512 height first to ensure correct dimensions
    if combined_image.height != 512:
        logger.info(
            "Resizing combined image from %sx%s to match 512px height",
            combined_image.width,
            combined_image.height,
        )
        aspect_ratio = combined_image.width / combined_image.height
        new_width = int(512 * aspect_ratio)
        combined_image = combined_image.resize((new_width, 512), Image.LANCZOS)

    # Determine output directory
    if create_subdirs:
        # Create subdirectory with same name as combined image (without extension)
        subdir_name = combined_image_path.stem
        output_path = Path(output_dir) / subdir_name
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Split and save images
    created_files = []
    x_offset = 0

    for i, (filename, width) in enumerate(zip(image_order, widths)):
        # Crop the section
        box = (x_offset, 0, x_offset + width, height)
        cropped_image = combined_image.crop(box)

        # Resize back to original resolution if available
        if original_resolutions and i < len(original_resolutions):
            orig_res = original_resolutions[i]
            orig_width = orig_res["width"]
            orig_height = orig_res["height"]
            cropped_image = cropped_image.resize(
                (orig_width, orig_height), Image.LANCZOS
            )
            logger.info(
                "  - %s: resized to %sx%s (original)", filename, orig_width, orig_height
            )
        else:
            logger.info(
                "  - %s: kept at %sx%s (no original resolution in metadata)",
                filename,
                width,
                height,
            )

        # Save the image
        output_file = output_path / filename
        cropped_image.save(output_file, quality=95)
        created_files.append(output_file)

        x_offset += width

    logger.info("Split %s into %d images", combined_image_path.name, len(created_files))
    return created_files


def process_all_combined_images(input_dir, output_dir, create_subdirs=True):
    """
    Process all combined images in the input directory.

    Args:
        input_dir: Directory containing combined images and metadata JSON files
        output_dir: Directory to save split images
        create_subdirs: If True, create subdirectories for each set of images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON metadata files
    json_files = list(input_path.glob("*.json"))

    if not json_files:
        logger.warning("No JSON metadata files found in %s", input_dir)
        return

    logger.info("Found %d metadata files to process", len(json_files))
    logger.info("Output directory: %s", output_dir)
    logger.info("Create subdirectories: %s", create_subdirs)

    successful = 0
    skipped = 0

    for json_file in sorted(json_files):
        try:
            # Find corresponding image file
            # Try common image extensions
            image_file = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = input_path / f"{json_file.stem}{ext}"
                if candidate.exists():
                    image_file = candidate
                    break

            if not image_file:
                logger.warning(
                    "Skipping %s: no corresponding image file found",
                    json_file.name,
                )
                skipped += 1
                continue

            # Split the image
            split_pane(image_file, json_file, output_path, create_subdirs)
            successful += 1

        except Exception as e:
            logger.exception("Error processing %s: %s", json_file.name, e)
            skipped += 1

    logger.info("=" * 50)
    logger.info("Processing complete!")
    logger.info("Successfully processed: %d", successful)
    logger.info("Skipped: %d", skipped)
    logger.info("=" * 50)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Split combined multi-pane images back into individual images"
    )
    parser.add_argument(
        "input_dir",
        help="Input directory containing combined images and metadata JSON files",
    )
    parser.add_argument("output_dir", help="Output directory for split images")
    parser.add_argument(
        "--no-subdirs",
        action="store_true",
        help="Save all images directly to output_dir without creating subdirectories",
    )

    args = parser.parse_args()

    process_all_combined_images(
        args.input_dir, args.output_dir, create_subdirs=not args.no_subdirs
    )


if __name__ == "__main__":
    main()
