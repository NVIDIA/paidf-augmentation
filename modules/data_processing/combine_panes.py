# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Combine 2–4+ images from each subdirectory into a single horizontal multi-pane image.
Resizes all images to 512px height while maintaining aspect ratio.
Saves metadata JSON for potential splitting later.
Skips only subdirectories with exactly 1 image; processes 2, 3, 4, or more images per ID.
"""

import json
import logging
from pathlib import Path
import argparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def resize_to_height(image, target_height=512):
    """Resize image to target height while maintaining aspect ratio."""
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)


def get_image_files(directory):
    """Get all image files from a directory, sorted alphabetically."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    image_files = []

    for file in sorted(Path(directory).iterdir()):
        if file.suffix.lower() in image_extensions:
            image_files.append(file)

    return image_files


def combine_pane(image_paths, output_path, metadata_path):
    """
    Combine 1 or more images into a single horizontal strip.

    Args:
        image_paths: List of 1+ image file paths (sorted)
        output_path: Path to save the combined image
        metadata_path: Path to save the metadata JSON
    """
    n = len(image_paths)
    if n < 1:
        raise ValueError(f"Need at least 1 image, got {n}")

    # Load and resize all images
    resized_images = []
    widths = []
    original_names = []
    original_resolutions = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Invalid image file: {img_path}")

        # Store original resolution
        height, width = img.shape[:2]
        original_resolutions.append({"width": width, "height": height})

        resized = resize_to_height(img, target_height=512)
        resized_images.append(resized)
        widths.append(resized.shape[1])
        original_names.append(img_path.name)

    # Calculate total width and create combined image
    total_width = sum(widths)
    combined_image = np.concatenate(resized_images, axis=1)

    # Save combined image
    if not cv2.imwrite(
        str(output_path), combined_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    ):
        raise OSError(f"Failed to write combined image: {output_path}")

    # Save metadata
    metadata = {
        "image_order": original_names,
        "widths": widths,
        "original_resolutions": original_resolutions,
        "total_width": total_width,
        "height": 512,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Created: %s", output_path.name)
    return metadata


def process_dataset(input_dir, output_dir):
    """
    Process all subdirectories in the input directory.
    Combines 2+ images per subdirectory into one horizontal image.
    Skips subdirectories with only 1 image.

    Args:
        input_dir: Root directory containing subdirectories (each with 2+ images)
        output_dir: Directory to save combined images and metadata
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all subdirectories
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not subdirs:
        logger.warning("No subdirectories found in %s", input_dir)
        return

    logger.info("Found %d subdirectories to process", len(subdirs))
    logger.info("Output directory: %s", output_dir)

    successful = 0
    skipped = 0

    for subdir in sorted(subdirs):
        processed_successfully = False
        skipped_subdir = False
        try:
            # Get all image files in subdirectory
            image_files = get_image_files(subdir)

            if len(image_files) == 0:
                logger.info("Skipping %s: no images found", subdir.name)
                skipped_subdir = True
                continue

            # Output filenames based on subdirectory name
            combined_image_path = output_path / f"{subdir.name}.jpg"
            metadata_path = output_path / f"{subdir.name}.json"

            # Combine images and save
            combine_pane(image_files, combined_image_path, metadata_path)
            processed_successfully = True

        except FileNotFoundError as e:
            logger.warning("Missing file while processing %s: %s", subdir.name, e)
            skipped_subdir = True
        except ValueError as e:
            logger.warning("Invalid input in %s: %s", subdir.name, e)
            skipped_subdir = True
        except OSError as e:
            logger.warning("I/O error while processing %s: %s", subdir.name, e)
            skipped_subdir = True
        except Exception as e:
            logger.exception("Unexpected error processing %s: %s", subdir.name, e)
            skipped_subdir = True
        finally:
            if processed_successfully:
                successful += 1
            elif skipped_subdir:
                skipped += 1

    logger.info("=" * 50)
    logger.info("Processing complete!")
    logger.info("Successfully processed: %d", successful)
    logger.info("Skipped: %d", skipped)
    logger.info("=" * 50)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Combine images from each subdirectory into horizontal multi-pane images"
    )
    parser.add_argument(
        "input_dir",
        help="Input directory containing subdirectories with images to combine",
    )
    parser.add_argument(
        "output_dir", help="Output directory for combined images and metadata"
    )

    args = parser.parse_args()

    process_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
