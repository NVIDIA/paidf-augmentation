# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import subprocess
import time

from typing import Dict, Optional, Tuple

import multistorageclient as msc


def generate_with_retries(
    generator,
    prompt: str,
    rgb_path: str,
    control_inputs: Dict[str, str],
    output_media_path: str,
    max_retries: int,
    backoff_base: float,
    logger: logging.Logger,
) -> Tuple[bool, Optional[str], float]:
    """Run generation, re-calling the endpoint on failure with exponential backoff.

    Each ``execute`` is one bounded request (the per-request timeout lives on the
    endpoint client), so a hung server fails after that timeout. On failure the
    endpoint is re-called up to ``max_retries`` times, waiting
    ``backoff_base * 2**n`` seconds between attempts (backoff_base, 2x, 4x, ...).
    The same request (same seed) is re-issued because the failure is the
    endpoint, not the produced output.

    Returns ``(success, output_path, elapsed_seconds)`` for the final attempt.
    """
    success = False
    output_path = None
    elapsed = 0.0
    for attempt in range(max_retries + 1):
        attempt_start = time.time()
        try:
            success, output_path = generator.execute(
                prompt, rgb_path, control_inputs, output_media_path
            )
        except Exception as e:
            success = False
            logger.error(f"Generation error -> {output_media_path}: {e}")
        elapsed = time.time() - attempt_start
        if success:
            return True, output_path, elapsed
        if attempt < max_retries:
            delay = backoff_base * (2**attempt)
            logger.warning(
                f"Generation failed (attempt {attempt + 1}/{max_retries + 1}); "
                f"retrying in {delay:.0f}s (exponential backoff)..."
            )
            time.sleep(delay)
    return False, output_path, elapsed


FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")
FFPROBE_TIMEOUT_S = 30

# pix_fmts NVDEC can decode (8-bit 4:2:0 family).
NVDEC_DECODABLE_PIX_FMTS = frozenset({"yuv420p", "yuvj420p", "nv12"})

# Case-insensitive H.264 ``profile`` substrings NVDEC cannot decode (4:4:4 /
# 4:2:2 chroma, 10-bit, Extended). A denylist, so decodable High variants like
# "Constrained High" / "Progressive High" are never falsely rejected.
UNSUPPORTED_H264_PROFILE_MARKERS = ("4:4:4", "4:2:2", "high 10", "extended")

# Codecs that decode only via NVDEC here (mpeg4/mjpeg have software decoders).
HW_ONLY_DECODE_CODECS = frozenset({"h264", "avc1"})


class UnsupportedVideoFormatError(RuntimeError):
    """Input video cannot be decoded by the container's ffmpeg."""


def pix_fmt_is_decodable(pix_fmt: Optional[str]) -> bool:
    """True if pix_fmt is NVDEC-decodable (8-bit 4:2:0); unknown/empty -> True."""
    if not pix_fmt:
        return True
    return pix_fmt.strip().lower() in NVDEC_DECODABLE_PIX_FMTS


def video_format_unsupported(
    codec_name: Optional[str], profile: Optional[str], pix_fmt: Optional[str]
) -> Optional[str]:
    """Return a reason string if the stream is undecodable here, else None.

    Only H.264 is gated; mpeg4/mjpeg ship software decoders.
    """
    if (codec_name or "").strip().lower() not in HW_ONLY_DECODE_CODECS:
        return None
    profile_l = (profile or "").strip().lower()
    if any(m in profile_l for m in UNSUPPORTED_H264_PROFILE_MARKERS):
        return f"H.264 profile {profile!r} (NVDEC decodes 8-bit 4:2:0 only)"
    if not pix_fmt_is_decodable(pix_fmt):
        return f"pixel format {pix_fmt!r} (NVDEC decodes 8-bit 4:2:0 only)"
    return None


# Container signatures, matched on leading bytes rather than extension: an
# endpoint's bytes land at whatever path the config named, so a WebM payload
# routinely arrives called ``.mp4``.
_CONTAINER_SIGNATURES = (
    (b"\x1a\x45\xdf\xa3", 0, "Matroska/WebM"),
    (b"ftyp", 4, "MP4/MOV"),
    (b"RIFF", 0, "AVI/RIFF"),
    (b"OggS", 0, "Ogg"),
    (b"FLV\x01", 0, "FLV"),
)

# The only container this image's FFmpeg can demux (build enables demuxer=mov).
SUPPORTED_CONTAINER = "MP4/MOV"


def sniff_container(path: str) -> Optional[str]:
    """Identify a media container from its magic bytes, or None if unrecognized."""
    try:
        with msc.open(path, "rb") as f:
            head = f.read(16)
    # Runs inside decode error handlers, so it must never throw and mask the
    # original failure; a remote read can fail with more than OSError.
    except Exception:
        return None
    for signature, offset, name in _CONTAINER_SIGNATURES:
        if head[offset : offset + len(signature)] == signature:
            return name
    return None


def decode_failure_hint(error_text: str, path: Optional[str] = None) -> Optional[str]:
    """Explain a decode failure caused by this image's restricted FFmpeg build.

    The build reports "cannot read that" the same way it reports a corrupt file,
    so return an actionable line for the known limits, or None otherwise.
    """
    if "cuvid" in error_text.lower():
        return (
            "H.264 decoding requires a GPU in this image (hardware h264_cuvid "
            "decoder; software AVC decode is disabled for licensing). Run the "
            "container with --gpus and NVIDIA_DRIVER_CAPABILITIES including 'video'."
        )
    container = sniff_container(path) if path else None
    if container and container != SUPPORTED_CONTAINER:
        return (
            f"Input is a {container} container, which this image's FFmpeg cannot "
            f"demux — only {SUPPORTED_CONTAINER} is enabled. The file extension is "
            "not the container: an endpoint returning VP9 in WebM produces this "
            "even when the path ends in .mp4. Remux to MP4 "
            "(ffmpeg -i in -c copy out.mp4) or configure the endpoint to return MP4."
        )
    return None


def probe_video_format(
    path: str, ffprobe_bin: str = FFPROBE_BIN, timeout: int = FFPROBE_TIMEOUT_S
) -> Dict[str, str]:
    """ffprobe the first video stream's codec_name/profile/pix_fmt.

    Returns those keys (string values), or {} when ffprobe is unavailable / the
    file has no video stream -- callers treat {} as "cannot tell, don't block".
    Metadata only (no ``-count_frames``), so probing never triggers a decode.
    """
    flags = "-v error -select_streams v:0 -show_entries stream=codec_name,profile,pix_fmt -of json"
    try:
        result = subprocess.run(  # noqa: S603 - intentional ffprobe call, shell=False
            [ffprobe_bin, *flags.split(), path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return {}  # ffprobe missing / cannot launch / timed out -> don't block

    # Parse whatever JSON ffprobe emitted, even on a non-zero exit: ``profile``
    # comes from the bitstream parser and survives a decoder-open failure.
    try:
        streams = (json.loads(result.stdout or "{}") or {}).get("streams") or []
    except (json.JSONDecodeError, TypeError):
        return {}
    if not streams:
        return {}
    s = streams[0]
    return {
        "codec_name": str(s.get("codec_name") or ""),
        "profile": str(s.get("profile") or ""),
        "pix_fmt": str(s.get("pix_fmt") or ""),
    }
