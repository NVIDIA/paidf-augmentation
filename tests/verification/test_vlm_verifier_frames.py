# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VLMVerifier frame sampling.

The verifier answers each MCQ against frames sampled from the generated clip, so
which frames it picks decides whether a mid-video event is ever seen. Decoding
goes through PyAV (OpenCV is built WITH_FFMPEG=OFF in this image), so ``av`` and
``cv2`` are stubbed here -- no real video, no ffmpeg, no GPU.
"""

import logging
from unittest.mock import patch

import pytest

from verification.vlm_verifier import VLMVerifier

LOGGER = logging.getLogger("test-vlm-verifier")


# ---------------------------------------------------------------------------
# PyAV doubles
# ---------------------------------------------------------------------------
class _Frame:
    def __init__(self, idx):
        self.idx = idx

    def to_ndarray(self, format=None):  # noqa: A002 - mirrors PyAV's kwarg
        return f"bgr-{self.idx}"


class _Stream:
    def __init__(self, frames):
        self.frames = frames  # 0 == container does not store nb_frames


class _Container:
    def __init__(self, n_real, declared_frames, has_video=True):
        self.n_real = n_real
        self.streams = type("S", (), {})()
        self.streams.video = [_Stream(declared_frames)] if has_video else []

    def decode(self, stream):
        for i in range(self.n_real):
            yield _Frame(i)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _av_open(n_real, declared_frames, has_video=True):
    """Return a stub av.open; a fresh container per call (_frame_count reopens)."""

    def opener(path, *a, **kw):
        return _Container(n_real, declared_frames, has_video)

    return opener


@pytest.fixture
def verifier():
    return VLMVerifier(
        system_prompt="Answer with one letter.",
        retry=0,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        max_tokens=10,
        stream=False,
        endpoint="http://localhost:9/v1",
        model="some/vlm",
        logger=LOGGER,
        frames=6,
    )


def _indices_from(paths):
    """frame_NN.jpg -> the NN ordinals actually written."""
    return [int(p.rsplit("_", 1)[1].split(".")[0]) for p in paths]


# ---------------------------------------------------------------------------
# _extract_frames
# ---------------------------------------------------------------------------
class TestExtractFrames:
    def test_single_frame_takes_only_the_first(self, verifier, tmp_path):
        written = []
        with (
            patch("verification.vlm_verifier.av.open", _av_open(93, 93)),
            patch(
                "verification.vlm_verifier.cv2.imwrite",
                side_effect=lambda p, img: written.append(img) or True,
            ),
        ):
            paths = verifier._extract_frames("clip.mp4", 1, str(tmp_path))
        assert len(paths) == 1
        assert written == ["bgr-0"]

    def test_six_frames_are_evenly_spaced_across_the_clip(self, verifier, tmp_path):
        """A mid-clip event must be visible, and the last frame must be included."""
        written = []
        with (
            patch("verification.vlm_verifier.av.open", _av_open(93, 93)),
            patch(
                "verification.vlm_verifier.cv2.imwrite",
                side_effect=lambda p, img: written.append(img) or True,
            ),
        ):
            paths = verifier._extract_frames("clip.mp4", 6, str(tmp_path))
        assert len(paths) == 6
        # round(i * 92 / 5) for i in 0..5
        assert written == [f"bgr-{i}" for i in (0, 18, 37, 55, 74, 92)]
        assert _indices_from(paths) == [0, 1, 2, 3, 4, 5]

    def test_missing_nb_frames_falls_back_to_counting_then_spaces_evenly(
        self, verifier, tmp_path
    ):
        """declared 0 -> _frame_count decodes to get the exact total."""
        written = []
        with (
            patch("verification.vlm_verifier.av.open", _av_open(9, 0)),
            patch(
                "verification.vlm_verifier.cv2.imwrite",
                side_effect=lambda p, img: written.append(img) or True,
            ),
        ):
            paths = verifier._extract_frames("clip.mp4", 3, str(tmp_path))
        assert len(paths) == 3
        # total=9 -> round(i * 8 / 2) = 0, 4, 8
        assert written == ["bgr-0", "bgr-4", "bgr-8"]

    def test_unknown_length_warns_and_takes_the_first_n(
        self, verifier, tmp_path, caplog
    ):
        """Count unavailable: sample the head, but say the spacing is not even."""

        def opener(path, *a, **kw):
            raise RuntimeError("cannot probe")

        container = _Container(5, 0)

        def first_ok_then_fail(path, *a, **kw):
            # _extract_frames' own open succeeds; _frame_count's reopen fails
            if getattr(first_ok_then_fail, "used", False):
                return opener(path)
            first_ok_then_fail.used = True
            return container

        with (
            caplog.at_level(logging.WARNING),
            patch("verification.vlm_verifier.av.open", first_ok_then_fail),
            patch("verification.vlm_verifier.cv2.imwrite", return_value=True),
        ):
            paths = verifier._extract_frames("clip.mp4", 3, str(tmp_path))
        assert len(paths) == 3
        assert any("Unknown frame count" in r.message for r in caplog.records)

    def test_no_video_stream_returns_empty(self, verifier, tmp_path, caplog):
        with (
            caplog.at_level(logging.ERROR),
            patch("verification.vlm_verifier.av.open", _av_open(0, 0, has_video=False)),
        ):
            paths = verifier._extract_frames("clip.mp4", 6, str(tmp_path))
        assert paths == []
        assert any("No video stream" in r.message for r in caplog.records)

    def test_failed_write_is_not_reported_as_an_extracted_frame(
        self, verifier, tmp_path, caplog
    ):
        """The caller opens every returned path, so a silent write failure would
        surface later as FileNotFoundError instead of triggering the fallback."""
        with (
            caplog.at_level(logging.ERROR),
            patch("verification.vlm_verifier.av.open", _av_open(93, 93)),
            patch("verification.vlm_verifier.cv2.imwrite", return_value=False),
        ):
            paths = verifier._extract_frames("clip.mp4", 6, str(tmp_path))
        assert paths == []
        assert any("Failed to write frame image" in r.message for r in caplog.records)

    def test_decode_error_returns_empty_and_logs_cause(
        self, verifier, tmp_path, caplog
    ):
        def boom(path, *a, **kw):
            raise RuntimeError("moov atom not found")

        with (
            caplog.at_level(logging.ERROR),
            patch("verification.vlm_verifier.av.open", boom),
        ):
            paths = verifier._extract_frames("clip.mp4", 6, str(tmp_path))
        assert paths == []
        assert any("moov atom not found" in r.message for r in caplog.records)

    def test_h264_without_gpu_gets_the_cuvid_hint(self, verifier, tmp_path, caplog):
        def boom(path, *a, **kw):
            raise RuntimeError("No decoder for h264_cuvid")

        with (
            caplog.at_level(logging.ERROR),
            patch("verification.vlm_verifier.av.open", boom),
        ):
            paths = verifier._extract_frames("clip.mp4", 6, str(tmp_path))
        assert paths == []
        assert any("--gpus" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _frame_count
# ---------------------------------------------------------------------------
class TestFrameCount:
    def test_uses_nb_frames_without_decoding(self):
        opened = []

        def opener(path, *a, **kw):
            opened.append(path)
            return _Container(93, 93)

        with patch("verification.vlm_verifier.av.open", opener):
            n = VLMVerifier._frame_count("clip.mp4", _Stream(93))
        assert n == 93
        assert opened == [], "nb_frames present -> must not reopen the container"

    def test_counts_by_decoding_when_nb_frames_absent(self):
        with patch("verification.vlm_verifier.av.open", _av_open(41, 0)):
            n = VLMVerifier._frame_count("clip.mp4", _Stream(0))
        assert n == 41

    def test_returns_zero_when_the_count_cannot_be_determined(self):
        def boom(path, *a, **kw):
            raise RuntimeError("unreadable")

        with patch("verification.vlm_verifier.av.open", boom):
            assert VLMVerifier._frame_count("clip.mp4", _Stream(0)) == 0


# ---------------------------------------------------------------------------
# _extract_first_frame (the fallback path)
# ---------------------------------------------------------------------------
class TestExtractFirstFrame:
    def test_success(self, verifier, tmp_path):
        out = str(tmp_path / "f.jpg")
        with (
            patch("verification.vlm_verifier.av.open", _av_open(10, 10)),
            patch("verification.vlm_verifier.cv2.imwrite", return_value=True),
        ):
            assert verifier._extract_first_frame("clip.mp4", out) is True

    def test_failed_write_reports_false(self, verifier, tmp_path, caplog):
        """Claiming success on a failed write hands the caller a path it will
        open and fail on -- this is already the fallback, so there is no net."""
        out = str(tmp_path / "f.jpg")
        with (
            caplog.at_level(logging.ERROR),
            patch("verification.vlm_verifier.av.open", _av_open(10, 10)),
            patch("verification.vlm_verifier.cv2.imwrite", return_value=False),
        ):
            assert verifier._extract_first_frame("clip.mp4", out) is False
        assert any("Failed to write first frame" in r.message for r in caplog.records)

    def test_no_video_stream_reports_false(self, verifier, tmp_path):
        out = str(tmp_path / "f.jpg")
        with patch(
            "verification.vlm_verifier.av.open", _av_open(0, 0, has_video=False)
        ):
            assert verifier._extract_first_frame("clip.mp4", out) is False

    def test_open_error_reports_false(self, verifier, tmp_path):
        def boom(path, *a, **kw):
            raise RuntimeError("bad file")

        out = str(tmp_path / "f.jpg")
        with patch("verification.vlm_verifier.av.open", boom):
            assert verifier._extract_first_frame("clip.mp4", out) is False
