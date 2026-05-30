# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for generation.utils -- the pre-decode input-format gate.

The slim container ffmpeg decodes H.264 only via NVDEC (h264_cuvid), which
handles 8-bit 4:2:0 only. A subtlety these tests pin down: under h264_cuvid
ffprobe reports pix_fmt as the decoder output surface (nv12) for *every* H.264
input, so the H.264 ``profile`` string is the dependable discriminator.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from generation.utils import (
    pix_fmt_is_decodable,
    probe_video_format,
    video_format_unsupported,
)


def _fake_ffprobe(streams):
    """Return a fake subprocess.run result whose stdout is ffprobe-style JSON."""
    return MagicMock(stdout=json.dumps({"streams": streams}), returncode=0)


# ---------------------------------------------------------------------------
# pix_fmt_is_decodable
# ---------------------------------------------------------------------------


class TestPixFmtIsDecodable:
    @pytest.mark.parametrize("pix_fmt", ["yuv420p", "yuvj420p", "nv12"])
    def test_supported_420_family(self, pix_fmt):
        assert pix_fmt_is_decodable(pix_fmt) is True

    @pytest.mark.parametrize("pix_fmt", ["YUV420P", " NV12 ", "Yuvj420p"])
    def test_case_and_whitespace_insensitive(self, pix_fmt):
        assert pix_fmt_is_decodable(pix_fmt) is True

    @pytest.mark.parametrize(
        "pix_fmt",
        ["yuv444p", "yuvj444p", "yuv422p", "yuv420p10le", "yuv444p10le", "p010le"],
    )
    def test_unsupported_chroma_or_bitdepth(self, pix_fmt):
        assert pix_fmt_is_decodable(pix_fmt) is False

    @pytest.mark.parametrize("pix_fmt", [None, ""])
    def test_unknown_does_not_block(self, pix_fmt):
        # We only block on a positive identification of an unsupported format.
        assert pix_fmt_is_decodable(pix_fmt) is True


# ---------------------------------------------------------------------------
# video_format_unsupported
# ---------------------------------------------------------------------------


class TestVideoFormatUnsupported:
    def test_real_slim_image_444_masked_as_nv12(self):
        # The crux: ffprobe in the slim image reports pix_fmt=nv12 for a true
        # 4:4:4 file, so detection MUST come from the profile string.
        reason = video_format_unsupported("h264", "High 4:4:4 Predictive", "nv12")
        assert reason is not None
        assert "4:4:4" in reason

    @pytest.mark.parametrize(
        "profile",
        [
            "High 4:4:4 Predictive",
            "High 4:4:4 Intra",
            "CAVLC 4:4:4 Intra",
            "High 4:2:2",
            "High 4:2:2 Intra",
            "High 10",
            "High 10 Intra",
            "Extended",  # XP profile (SP/SI slices) -- NVDEC cannot decode it
        ],
    )
    def test_unsupported_h264_profiles(self, profile):
        # pix_fmt is the (useless) nv12 the hw decoder reports -- profile decides.
        assert video_format_unsupported("h264", profile, "nv12") is not None

    @pytest.mark.parametrize(
        "profile",
        [
            "Baseline",
            "Constrained Baseline",
            "Main",
            "High",
            # 8-bit 4:2:0 High-profile variants NVDEC decodes fine -- a
            # conservative allowlist would wrongly reject these, so the denylist
            # must let them through.
            "Constrained High",
            "Progressive High",
        ],
    )
    def test_supported_h264_profiles(self, profile):
        assert video_format_unsupported("h264", profile, "nv12") is None

    def test_pix_fmt_signal_when_profile_blank(self):
        # Richer ffmpeg builds (and unit mocks) report the true bitstream pix_fmt.
        assert video_format_unsupported("h264", "", "yuv444p") is not None
        assert video_format_unsupported("h264", None, "yuv422p") is not None
        assert video_format_unsupported("h264", "", "yuv420p10le") is not None

    def test_supported_h264_with_true_420_pix_fmt(self):
        assert video_format_unsupported("h264", "Main", "yuv420p") is None

    def test_avc1_alias_treated_as_h264(self):
        assert (
            video_format_unsupported("avc1", "High 4:4:4 Predictive", "nv12")
            is not None
        )

    def test_case_insensitive(self):
        assert (
            video_format_unsupported("H264", "HIGH 4:4:4 PREDICTIVE", "NV12")
            is not None
        )

    @pytest.mark.parametrize(
        ("codec", "profile", "pix_fmt"),
        [
            ("mjpeg", "", "yuvj444p"),  # mjpeg has a software decoder; 4:4:4 OK
            ("mjpeg", "", "yuvj422p"),
            ("mpeg4", "Simple Profile", "yuv420p"),
            ("vp9", "Profile 1", "yuv444p"),  # not a hw-only codec here
        ],
    )
    def test_non_h264_codecs_not_gated(self, codec, profile, pix_fmt):
        assert video_format_unsupported(codec, profile, pix_fmt) is None

    def test_all_none_does_not_block(self):
        assert video_format_unsupported(None, None, None) is None


# ---------------------------------------------------------------------------
# probe_video_format
# ---------------------------------------------------------------------------


class TestProbeVideoFormat:
    def test_parses_first_stream(self):
        streams = [
            {
                "codec_name": "h264",
                "profile": "High 4:4:4 Predictive",
                "pix_fmt": "nv12",
            }
        ]
        with patch.object(
            subprocess, "run", return_value=_fake_ffprobe(streams)
        ) as run:
            fmt = probe_video_format("/data/in.mp4")
        assert fmt == {
            "codec_name": "h264",
            "profile": "High 4:4:4 Predictive",
            "pix_fmt": "nv12",
        }
        # Must NOT force a decode (no -count_frames), or it would hang on 4:4:4.
        assert "-count_frames" not in run.call_args[0][0]

    def test_missing_fields_default_to_empty_string(self):
        with patch.object(
            subprocess, "run", return_value=_fake_ffprobe([{"codec_name": "h264"}])
        ):
            fmt = probe_video_format("/data/in.mp4")
        assert fmt == {"codec_name": "h264", "profile": "", "pix_fmt": ""}

    def test_no_streams_returns_empty(self):
        with patch.object(subprocess, "run", return_value=_fake_ffprobe([])):
            assert probe_video_format("/data/in.mp4") == {}

    def test_no_streams_key_returns_empty(self):
        with patch.object(subprocess, "run", return_value=MagicMock(stdout="{}")):
            assert probe_video_format("/data/in.mp4") == {}

    def test_invalid_json_returns_empty(self):
        with patch.object(subprocess, "run", return_value=MagicMock(stdout="not json")):
            assert probe_video_format("/data/in.mp4") == {}

    @pytest.mark.parametrize(
        "exc",
        [
            FileNotFoundError("ffprobe not found"),  # ffprobe binary absent
            subprocess.TimeoutExpired("ffprobe", 30),  # probe hung
        ],
    )
    def test_probe_launch_failure_returns_empty(self, exc):
        with patch.object(subprocess, "run", side_effect=exc):
            assert probe_video_format("/data/in.mp4") == {}

    def test_nonzero_exit_with_valid_json_is_still_used(self):
        # The profile is produced by the bitstream parser, so honor complete
        # JSON even when ffprobe exits non-zero (e.g. a decoder-open warning).
        streams = [
            {
                "codec_name": "h264",
                "profile": "High 4:4:4 Predictive",
                "pix_fmt": "nv12",
            }
        ]
        result = MagicMock(stdout=json.dumps({"streams": streams}), returncode=1)
        with patch.object(subprocess, "run", return_value=result):
            fmt = probe_video_format("/data/in.mp4")
        assert fmt["profile"] == "High 4:4:4 Predictive"

    def test_nonzero_exit_with_partial_json_returns_empty(self):
        # When the decoder cannot open at all, ffprobe emits truncated JSON.
        result = MagicMock(stdout="{", returncode=1)
        with patch.object(subprocess, "run", return_value=result):
            assert probe_video_format("/data/in.mp4") == {}
