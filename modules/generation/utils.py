# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess

from typing import Dict, Optional

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
