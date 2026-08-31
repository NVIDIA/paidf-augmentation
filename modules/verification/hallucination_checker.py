# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hallucination checker for video augmentation.

This module implements a computer vision-based hallucination detection algorithm
that compares temporal dynamics (motion) between original and augmented videos
to detect hallucinated motion artifacts.
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Dict, Tuple

import av
import cv2
import multistorageclient as msc
import numpy as np


@dataclass
class HallucinationParams:
    """Parameters for hallucination detection."""

    grad_thresh: float = 10.0
    blur_ksize: int = 7
    morph_k: int = 3
    dist_tol_px: float = 7.0
    max_frames: int | None = None


class HallucinationChecker:
    """
    Checks for hallucinated motion in augmented videos.

    Compares temporal dynamics between original and augmented videos
    to detect motion artifacts not present in the original video.

    Algorithm:
    1. For each frame pair (original, augmented):
       - Compute grayscale difference from previous frame
       - Apply Gaussian blur and threshold to create dynamic mask
       - Apply morphological opening to reduce noise
    2. Use distance transforms to identify hallucinated pixels:
       - Augmented dynamic pixels > dist_tol_px from any original dynamic pixel
    3. Calculate score: 1.0 - (hallucinated_pixels / total_augmented_dynamic_pixels)
    """

    def __init__(self, params: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the HallucinationChecker.

        Args:
            params: Dictionary of hallucination detection parameters
            logger: Logger instance
        """
        self.params = HallucinationParams(
            grad_thresh=float(params.get("grad_thresh", 10.0)),
            blur_ksize=int(params.get("blur_ksize", 7)),
            morph_k=int(params.get("morph_k", 3)),
            dist_tol_px=float(params.get("dist_tol_px", 7.0)),
            max_frames=int(params["max_frames"])
            if params.get("max_frames") is not None
            else None,
        )
        self.logger = logger

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale."""
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _ensure_same_size(
        self, frame: np.ndarray, target_hw: Tuple[int, int]
    ) -> np.ndarray:
        """Resize frame to match target dimensions."""
        th, tw = target_hw
        h, w = frame.shape[:2]
        if (h, w) == (th, tw):
            return frame
        return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

    def _compute_dynamic_mask(
        self,
        prev_gray: np.ndarray | None,
        curr_gray: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dynamic mask from temporal gradient.

        Args:
            prev_gray: Previous frame in grayscale (or None for first frame)
            curr_gray: Current frame in grayscale

        Returns:
            Tuple of (dynamic_mask, curr_gray) where dynamic_mask is binary
        """
        if prev_gray is None:
            return np.zeros_like(curr_gray, dtype=np.uint8), curr_gray

        # Compute absolute difference
        diff = cv2.absdiff(curr_gray, prev_gray)

        # Apply Gaussian blur
        k = max(1, self.params.blur_ksize | 1)  # Ensure odd
        if k >= 3:
            diff = cv2.GaussianBlur(diff, (k, k), 0)

        # Threshold to create binary mask
        _, mask = cv2.threshold(diff, self.params.grad_thresh, 255, cv2.THRESH_BINARY)

        # Apply morphological opening to reduce noise
        mk = max(1, self.params.morph_k | 1)  # Ensure odd
        if mk >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask, curr_gray

    def _hallucination_counts(
        self,
        orig_mask: np.ndarray,
        aug_mask: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Count hallucinated pixels.

        Args:
            orig_mask: Binary dynamic mask from original video
            aug_mask: Binary dynamic mask from augmented video

        Returns:
            Tuple of (hallucinated_count, total_augmented_dynamic_count)
        """
        orig_dyn = orig_mask > 0
        aug_dyn = aug_mask > 0
        num_aug = int(np.count_nonzero(aug_dyn))

        if num_aug == 0:
            return 0, 0

        # Compute distance transform from original dynamic regions
        src = np.where(orig_dyn, 0, 255).astype(np.uint8)
        dist = cv2.distanceTransform(src, cv2.DIST_L2, 3)

        # Identify hallucinated pixels (augmented dynamics far from original dynamics)
        hallucinated = (aug_dyn) & (dist > float(self.params.dist_tol_px))

        return int(np.count_nonzero(hallucinated)), num_aug

    def _download_if_remote(self, path: str) -> str:
        """
        Download video to temporary file if it's a remote path.

        Args:
            path: Video path (local or remote)

        Returns:
            Local path to video file
        """
        # Check if path is remote (starts with s3://, gs://, etc.)
        if path.startswith(
            ("s3://", "gs://", "azure://", "msc://", "https://", "http://")
        ):
            self.logger.debug(f"Downloading remote video: {path}")
            # Create temp file
            suffix = os.path.splitext(path)[1] or ".mp4"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_path = temp_file.name
            temp_file.close()

            # Download using multistorageclient
            with msc.open(path, "rb") as src:
                with open(temp_path, "wb") as dst:
                    dst.write(src.read())

            self.logger.debug(f"Downloaded to: {temp_path}")
            return temp_path
        else:
            return path

    def check_hallucination(
        self,
        original_video_path: str,
        augmented_video_path: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for hallucinated motion in augmented video.

        Args:
            original_video_path: Path to original video
            augmented_video_path: Path to augmented video

        Returns:
            Tuple of (passed: bool, details: dict)

        The details dictionary contains:
            - "score": Overall hallucination score [0, 1]
            - "total_frames": Number of frames processed
            - "total_hallucinated_pixels": Total hallucinated pixels
            - "total_augmented_dynamic_pixels": Total dynamic pixels in augmented video
            - "params": Detection parameters used
        """
        self.logger.info("Starting hallucination check...")
        self.logger.debug(f"Original video: {original_video_path}")
        self.logger.debug(f"Augmented video: {augmented_video_path}")

        start_time = time.time()

        temp_files = []

        try:
            # Download videos if remote
            orig_path = self._download_if_remote(original_video_path)
            aug_path = self._download_if_remote(augmented_video_path)

            if orig_path != original_video_path:
                temp_files.append(orig_path)
            if aug_path != augmented_video_path:
                temp_files.append(aug_path)

            with av.open(orig_path) as cont_o, av.open(aug_path) as cont_a:
                if not cont_o.streams.video or not cont_a.streams.video:
                    error_msg = (
                        f"No video stream in inputs: orig={orig_path}, aug={aug_path}"
                    )
                    self.logger.error(error_msg)
                    return False, {
                        "score": 0.0,
                        "error": error_msg,
                        "params": vars(self.params),
                    }

                it_o = (
                    f.to_ndarray(format="bgr24")
                    for f in cont_o.decode(cont_o.streams.video[0])
                )
                it_a = (
                    f.to_ndarray(format="bgr24")
                    for f in cont_a.decode(cont_a.streams.video[0])
                )

                fo = next(it_o, None)
                fa = next(it_a, None)

                if fo is None or fa is None:
                    error_msg = "Empty videos or failed to read first frames"
                    self.logger.error(error_msg)
                    return False, {
                        "score": 0.0,
                        "error": error_msg,
                        "total_frames": 0,
                        "params": vars(self.params),
                    }

                h, w = fo.shape[:2]
                fa = self._ensure_same_size(fa, (h, w))
                prev_o = self._to_gray(fo)
                prev_a = self._to_gray(fa)

                total_h = 0
                total_aug = 0
                frame_count = 1

                # The generated video may not have exactly the same frame count as
                # the input (e.g. Cosmos Transfer can emit a few extra frames), so
                # align frame-by-frame over the overlap rather than requiring equal
                # lengths. ``zip_longest`` with a sentinel stops at the shorter
                # stream *without* silently pulling and dropping a surplus frame
                # from the longer one (plain ``zip`` discards the extra ``it_o``
                # frame it consumes when ``it_a`` ends first).
                frame_missing = object()
                for fo, fa in zip_longest(it_o, it_a, fillvalue=frame_missing):
                    if fo is frame_missing or fa is frame_missing:
                        # Length mismatch: this pair holds the first surplus frame
                        # of the longer stream (the other side is already
                        # exhausted). Count it plus the remaining tail — without
                        # comparing — so no frame is consumed silently, then stop.
                        extra_o = (0 if fo is frame_missing else 1) + sum(
                            1 for _ in it_o
                        )
                        extra_a = (0 if fa is frame_missing else 1) + sum(
                            1 for _ in it_a
                        )
                        if extra_o or extra_a:
                            self.logger.warning(
                                "Hallucination check: input/output frame counts differ "
                                f"(compared {frame_count} aligned frames; "
                                f"{extra_o} extra input + {extra_a} extra output frame(s) ignored)."
                            )
                        break

                    fa = self._ensure_same_size(fa, (h, w))
                    go = self._to_gray(fo)
                    ga = self._to_gray(fa)

                    mo, prev_o = self._compute_dynamic_mask(prev_o, go)
                    ma, prev_a = self._compute_dynamic_mask(prev_a, ga)

                    nh, nau = self._hallucination_counts(mo, ma)
                    total_h += nh
                    total_aug += nau

                    frame_count += 1

                    if (
                        self.params.max_frames is not None
                        and frame_count >= self.params.max_frames
                    ):
                        self.logger.debug(
                            f"Reached max_frames limit: {self.params.max_frames}"
                        )
                        break

            # Calculate score
            if total_aug == 0:
                score = 1.0
            else:
                score = max(0.0, 1.0 - (float(total_h) / float(total_aug)))

            self.logger.info(f"Hallucination check complete: score={score:.4f}")
            self.logger.debug(f"Frames processed: {frame_count}")
            self.logger.debug(f"Hallucinated pixels: {total_h} / {total_aug}")

            elapsed_seconds = time.time() - start_time

            details = {
                "score": float(score),
                "total_frames": frame_count,
                "total_hallucinated_pixels": total_h,
                "total_augmented_dynamic_pixels": total_aug,
                "duration_seconds": elapsed_seconds,
                "params": vars(self.params),
            }

            return True, details

        except Exception as e:
            self.logger.error(f"Error during hallucination check: {e}", exc_info=True)
            return False, {
                "score": 0.0,
                "error": str(e),
                "params": vars(self.params),
            }

        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        self.logger.debug(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
