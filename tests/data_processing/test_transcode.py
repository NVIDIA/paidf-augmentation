# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the VP9 transcode post-processor.

``transcode_video`` overwrites ``data.output.video`` in place, so the branch that
matters most is the backup/restore rollback: it is the only thing standing
between a failed copy and a destroyed generation. Everything here runs without
ffmpeg or a GPU -- ``subprocess.run`` and ``probe_video_format`` are stubbed.
"""

import logging
import os
import subprocess
import tempfile
from unittest.mock import patch

import pytest

from data_processing.transcode import transcode_video

LOGGER = logging.getLogger("test-transcode")

ORIGINAL = b"ORIGINAL-GENERATED-VIDEO"
ENCODED = b"RE-ENCODED-VP9-BYTES"


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def dest(workdir):
    """An existing destination holding the 'generated' video."""
    p = os.path.join(workdir, "out.mp4")
    with open(p, "wb") as f:
        f.write(ORIGINAL)
    return p


def _fake_ffmpeg(payload=ENCODED, returncode=0, stderr=""):
    """Stub subprocess.run: write ``payload`` to the ffmpeg output path."""

    def run(cmd, **kwargs):
        if returncode == 0:
            with open(cmd[-1], "wb") as f:
                f.write(payload)
        return subprocess.CompletedProcess(cmd, returncode, "", stderr)

    return run


def _probe(*codecs):
    """Stub probe_video_format returning each codec in turn (source, then output)."""
    seq = list(codecs)

    def probe(path):
        val = seq.pop(0) if seq else None
        return {"codec_name": val} if val is not None else {}

    return probe


# ---------------------------------------------------------------------------
# Guard rails: things that must never invoke ffmpeg
# ---------------------------------------------------------------------------
class TestSkips:
    def test_unsupported_codec_reports_instead_of_raising(self, dest):
        with patch("data_processing.transcode.subprocess.run") as run:
            res = transcode_video(dest, dest, {"codec": "av1"}, LOGGER)
        assert res["transcoded"] is False
        assert "unsupported target codec" in res["error"]
        run.assert_not_called()
        # destination untouched
        assert open(dest, "rb").read() == ORIGINAL

    @pytest.mark.parametrize("name", ["out.jpg", "out.png"])
    def test_image_output_is_skipped(self, workdir, name):
        p = os.path.join(workdir, name)
        with open(p, "wb") as f:
            f.write(ORIGINAL)
        with patch("data_processing.transcode.subprocess.run") as run:
            res = transcode_video(p, p, {"codec": "vp9"}, LOGGER)
        assert res["skipped_reason"] == "image_output"
        assert res["transcoded"] is False
        run.assert_not_called()
        assert open(p, "rb").read() == ORIGINAL

    def test_already_vp9_is_left_byte_identical(self, dest):
        with (
            patch("data_processing.transcode.probe_video_format", _probe("vp9")),
            patch("data_processing.transcode.subprocess.run") as run,
        ):
            res = transcode_video(dest, dest, {"codec": "vp9"}, LOGGER)
        assert res["skipped_reason"] == "already_target_codec"
        assert res["transcoded"] is False
        assert res["source_codec"] == "vp9"
        run.assert_not_called()
        assert open(dest, "rb").read() == ORIGINAL

    def test_force_re_encodes_even_when_already_vp9(self, dest):
        with (
            patch("data_processing.transcode.probe_video_format", _probe("vp9", "vp9")),
            patch("data_processing.transcode.subprocess.run", _fake_ffmpeg()),
        ):
            res = transcode_video(dest, dest, {"codec": "vp9", "force": True}, LOGGER)
        assert res["transcoded"] is True
        assert open(dest, "rb").read() == ENCODED


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class TestSuccess:
    def test_h264_to_vp9_replaces_destination(self, dest):
        with (
            patch(
                "data_processing.transcode.probe_video_format", _probe("h264", "vp9")
            ),
            patch("data_processing.transcode.subprocess.run", _fake_ffmpeg()),
        ):
            res = transcode_video(dest, dest, {"codec": "vp9"}, LOGGER)
        assert res["transcoded"] is True
        assert res["source_codec"] == "h264"
        assert res["output_bytes"] == len(ENCODED)
        assert "error" not in res
        assert open(dest, "rb").read() == ENCODED

    def test_encoder_flags_are_libvpx_crf_mode(self, dest):
        seen = {}

        def run(cmd, **kwargs):
            seen["cmd"] = cmd
            with open(cmd[-1], "wb") as f:
                f.write(ENCODED)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with (
            patch(
                "data_processing.transcode.probe_video_format", _probe("h264", "vp9")
            ),
            patch("data_processing.transcode.subprocess.run", run),
        ):
            transcode_video(dest, dest, {"codec": "vp9", "crf": 28, "speed": 2}, LOGGER)

        cmd = seen["cmd"]
        assert "libvpx-vp9" in cmd
        assert cmd[cmd.index("-crf") + 1] == "28"
        assert cmd[cmd.index("-cpu-used") + 1] == "2"
        # -b:v 0 is required for libvpx-vp9 constant-quality mode
        assert cmd[cmd.index("-b:v") + 1] == "0"
        assert "-an" in cmd  # audio dropped

    def test_unreadable_probe_of_output_does_not_discard_a_good_encode(self, dest):
        """probe_video_format returns {} for 'cannot tell' (no ffprobe / timeout).

        That is not evidence the encode failed; treating it as failure deleted a
        perfectly good VP9 file and reported an error despite ffmpeg exiting 0.
        """
        with (
            patch("data_processing.transcode.probe_video_format", _probe("h264", None)),
            patch("data_processing.transcode.subprocess.run", _fake_ffmpeg()),
        ):
            res = transcode_video(dest, dest, {"codec": "vp9"}, LOGGER)
        assert res["transcoded"] is True
        assert "error" not in res
        assert open(dest, "rb").read() == ENCODED


# ---------------------------------------------------------------------------
# Failure modes: the generation must survive every one of them
# ---------------------------------------------------------------------------
class TestFailuresPreserveOriginal:
    def test_ffmpeg_nonzero_exit(self, dest):
        with (
            patch("data_processing.transcode.probe_video_format", _probe("h264")),
            patch(
                "data_processing.transcode.subprocess.run",
                _fake_ffmpeg(returncode=1, stderr="boom"),
            ),
        ):
            res = transcode_video(dest, dest, {"codec": "vp9"}, LOGGER)
        assert res["transcoded"] is False
        assert "ffmpeg exited 1" in res["error"]
        assert open(dest, "rb").read() == ORIGINAL

    def test_ffmpeg_timeout(self, dest):
        def run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 1)

        with (
            patch("data_processing.transcode.probe_video_format", _probe("h264")),
            patch("data_processing.transcode.subprocess.run", run),
        ):
            res = transcode_video(dest, dest, {"codec": "vp9"}, LOGGER)
        assert res["transcoded"] is False
        assert "timeout" in res["error"]
        assert open(dest, "rb").read() == ORIGINAL

    def test_empty_output_file(self, dest):
        with (
            patch("data_processing.transcode.probe_video_format", _probe("h264")),
            patch(
                "data_processing.transcode.subprocess.run", _fake_ffmpeg(payload=b"")
            ),
        ):
            res = transcode_video(dest, dest, {"codec": "vp9"}, LOGGER)
        assert res["transcoded"] is False
        assert "empty file" in res["error"]
        assert open(dest, "rb").read() == ORIGINAL

    def test_wrong_codec_written(self, dest):
        with (
            patch(
                "data_processing.transcode.probe_video_format", _probe("h264", "h264")
            ),
            patch("data_processing.transcode.subprocess.run", _fake_ffmpeg()),
        ):
            res = transcode_video(dest, dest, {"codec": "vp9"}, LOGGER)
        assert res["transcoded"] is False
        assert "expected vp9" in res["error"]
        assert open(dest, "rb").read() == ORIGINAL

    def test_h264_without_gpu_logs_cuvid_hint(self, dest, caplog):
        with (
            caplog.at_level(logging.ERROR),
            patch("data_processing.transcode.probe_video_format", _probe("h264")),
            patch(
                "data_processing.transcode.subprocess.run",
                _fake_ffmpeg(returncode=1, stderr="Cannot load h264_cuvid"),
            ),
        ):
            res = transcode_video(dest, dest, {"codec": "vp9"}, LOGGER)
        assert res["transcoded"] is False
        assert any("--gpus" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Rollback -- the branch that protects the generated video
# ---------------------------------------------------------------------------
class TestRollback:
    def test_failed_write_restores_the_original(self, dest):
        """A copy that dies midway must leave the generation intact."""
        real_open = open
        calls = {"n": 0}

        def flaky_msc_open(path, mode="rb", **kw):
            # 1st write = the destructive copy (fail it), 2nd = the restore.
            if "w" in mode:
                calls["n"] += 1
                if calls["n"] == 1:
                    with real_open(path, "wb") as f:
                        f.write(b"TRUNC")  # simulate partial write before the error
                    raise OSError("disk full")
            return real_open(path, mode)

        with (
            patch(
                "data_processing.transcode.probe_video_format", _probe("h264", "vp9")
            ),
            patch("data_processing.transcode.subprocess.run", _fake_ffmpeg()),
            patch("data_processing.transcode.msc.is_file", return_value=True),
            patch("data_processing.transcode.msc.open", flaky_msc_open),
        ):
            res = transcode_video(dest, dest, {"codec": "vp9"}, LOGGER)

        assert res["transcoded"] is False
        assert "disk full" in res["error"]
        assert "recovery_copy" not in res
        # the generated video is back, not the truncated stub
        assert open(dest, "rb").read() == ORIGINAL

    def test_failed_restore_keeps_backup_and_reports_its_path(self, dest):
        """If the restore also fails, the last intact copy must not be deleted."""
        real_open = open

        def always_failing_write(path, mode="rb", **kw):
            if "w" in mode:
                raise OSError("storage gone")
            return real_open(path, mode)

        with (
            patch(
                "data_processing.transcode.probe_video_format", _probe("h264", "vp9")
            ),
            patch("data_processing.transcode.subprocess.run", _fake_ffmpeg()),
            patch("data_processing.transcode.msc.is_file", return_value=True),
            patch("data_processing.transcode.msc.open", always_failing_write),
        ):
            res = transcode_video(dest, dest, {"codec": "vp9"}, LOGGER)

        assert res["transcoded"] is False
        kept = res.get("recovery_copy")
        assert kept, "recovery_copy must name the surviving backup"
        # the finally-block must NOT have deleted the only intact copy
        assert os.path.exists(kept)
        assert open(kept, "rb").read() == ORIGINAL
        os.unlink(kept)

    def test_missing_destination_skips_backup(self, workdir):
        """Nothing to preserve -> no backup, and the write still happens."""
        dest = os.path.join(workdir, "new.mp4")
        src = os.path.join(workdir, "src.mp4")
        with open(src, "wb") as f:
            f.write(ORIGINAL)
        with (
            patch(
                "data_processing.transcode.probe_video_format", _probe("h264", "vp9")
            ),
            patch("data_processing.transcode.subprocess.run", _fake_ffmpeg()),
        ):
            res = transcode_video(src, dest, {"codec": "vp9"}, LOGGER)
        assert res["transcoded"] is True
        assert open(dest, "rb").read() == ENCODED

    def test_remote_dest_lookup_error_aborts_before_destructive_write(self, dest):
        """A transient storage fault must not be read as 'nothing to preserve'."""
        with (
            patch(
                "data_processing.transcode.probe_video_format", _probe("h264", "vp9")
            ),
            patch("data_processing.transcode.subprocess.run", _fake_ffmpeg()),
            patch(
                "data_processing.transcode.msc.is_file", side_effect=OSError("s3 down")
            ),
            patch("data_processing.transcode.msc.open") as msc_open,
        ):
            res = transcode_video(dest, "s3://bucket/out.mp4", {"codec": "vp9"}, LOGGER)
        assert res["transcoded"] is False
        assert "s3 down" in res["error"]
        msc_open.assert_not_called()  # never reached the destructive copy


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
def test_temp_files_are_cleaned_up(dest):
    created = []
    real_ntf = tempfile.NamedTemporaryFile

    def tracking_ntf(*a, **kw):
        fh = real_ntf(*a, **kw)
        created.append(fh.name)
        return fh

    with (
        patch("data_processing.transcode.probe_video_format", _probe("h264", "vp9")),
        patch("data_processing.transcode.subprocess.run", _fake_ffmpeg()),
        patch("data_processing.transcode.tempfile.NamedTemporaryFile", tracking_ntf),
    ):
        res = transcode_video(dest, dest, {"codec": "vp9"}, LOGGER)

    assert res["transcoded"] is True
    assert created, "expected at least the ffmpeg output temp file"
    for p in created:
        assert not os.path.exists(p), f"temp file leaked: {p}"
