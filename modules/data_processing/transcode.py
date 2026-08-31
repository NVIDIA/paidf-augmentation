# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize a generated video to a single output codec (VP9).

Augmentation writes back whatever bitstream the model endpoint returned, so the
output codec varies by model (the Cosmos Transfer NIM emits VP9, vLLM-omni emits
H.264). Downstream containers -- and datasets that mix augmented clips with UAL
artifacts -- expect one format. This post-processor re-encodes the output to VP9
so the pipeline emits a uniform codec regardless of which model produced it.

A source that is already the target codec is left byte-identical unless
``force`` is set, so the common Cosmos Transfer path costs nothing and takes no
generation loss.

Image outputs are skipped outright: ``data.output.video`` also carries still
images for the image-edit model, and re-encoding a JPG/PNG into a one-frame MP4
would corrupt it for every downstream reader.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional

import multistorageclient as msc

from aug_utils.constants import is_image
from generation.utils import decode_failure_hint, probe_video_format

# Union of the remote schemes handled elsewhere in the repo (``REMOTE_PATH_PREFIXES``
# in create_attribute_augmented_dataset, and the inline tuple in hallucination_checker);
# each of those omits schemes the other lists, so cover both.
_REMOTE_PREFIXES = (
    "s3://",
    "gs://",
    "az://",
    "azure://",
    "msc://",
    "https://",
    "http://",
)

# Target codec -> (ffmpeg encoder, stream codec_name reported by ffprobe).
_ENCODERS = {"vp9": ("libvpx-vp9", "vp9")}

# Mirror generation.utils' FFPROBE_BIN convention so both binaries can be
# redirected together when the OSRB-cleared build is not first on PATH.
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


def _timeout_from_env(default: int = 3600) -> int:
    """Encoding must not wedge the pipeline, but a typo must not kill it either.

    Parsed defensively: this module is imported lazily from inside the sample
    loop, so a ValueError here would abort a run whose generation and evaluation
    have already been paid for.
    """
    raw = os.environ.get("TRANSCODE_TIMEOUT_S")
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


_FFMPEG_TIMEOUT_S = _timeout_from_env()


def _encode_threads() -> int:
    """Threads for libvpx-vp9, honoring CPU affinity.

    ``os.cpu_count()`` reports host cores and ignores a cgroup CPU quota, which
    would oversubscribe libvpx inside a ``--cpus``-limited container.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 4)


def _is_remote(path: str) -> bool:
    return path.startswith(_REMOTE_PREFIXES)


def _dest_exists(path: str) -> bool:
    """Whether there is an existing object at ``path`` worth backing up.

    Checked explicitly rather than by catching the read error, so a transient
    storage fault cannot be mistaken for "nothing to preserve" and silently
    disable the rollback before the destructive write.

    A failed lookup is not evidence of absence. For a remote URI it is far more
    likely a transient fault, and ``os.path.exists`` on a URI always returns
    False -- which would skip the backup and strip rollback protection right
    before the destructive write. So propagate it and let the caller abort with
    the destination still intact.
    """
    try:
        return bool(msc.is_file(path))
    except Exception:
        if _is_remote(path):
            raise
        return os.path.exists(path)


def transcode_video(
    src_path: str,
    dest_path: str,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Re-encode ``src_path`` to the configured codec and write to ``dest_path``.

    Returns a metadata dict describing what happened. Never raises: on any
    failure the original output is left in place and the reason is reported via
    the returned ``error`` field, because the generated video is still valid
    even when normalization could not run.

    NOTE: callers run this after the evaluators, so when ``transcoded`` is true
    the shipped bitstream is a re-encode of the one that was evaluated.
    """
    result: Dict[str, Any] = {"transcoded": False}

    codec = (config.get("codec") or "vp9").lower()
    result["target_codec"] = codec
    if codec not in _ENCODERS:
        # Keep the "never raises" contract: report instead of KeyError.
        msg = f"unsupported target codec {codec!r}; supported: {sorted(_ENCODERS)}"
        logger.error(f"Transcode skipped: {msg}")
        result["error"] = msg
        return result
    encoder, target_stream_codec = _ENCODERS[codec]

    # ``output.video`` doubles as the image-edit output path; never rewrap a still.
    if is_image(dest_path) or is_image(src_path):
        logger.info(f"Transcode skipped: {dest_path} is an image, not a video")
        result["skipped_reason"] = "image_output"
        return result

    tmp_in: Optional[str] = None
    tmp_out: Optional[str] = None
    backup: Optional[str] = None

    try:
        # ffmpeg/ffprobe need a local file; pull remote inputs down first.
        if _is_remote(src_path):
            fh = tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(src_path)[1] or ".mp4"
            )
            tmp_in = fh.name
            fh.close()
            with msc.open(src_path, "rb") as s, open(tmp_in, "wb") as d:
                shutil.copyfileobj(s, d)
            local_src = tmp_in
        else:
            local_src = src_path

        # Reuse the shared probe: it has a timeout, honors FFPROBE_BIN, and keeps
        # whatever JSON ffprobe emitted even on a non-zero exit.
        source_codec = (probe_video_format(local_src).get("codec_name") or "").lower()
        result["source_codec"] = source_codec or None

        if source_codec == target_stream_codec and not config.get("force"):
            logger.info(
                f"Transcode skipped: output is already {codec} "
                f"(set data_processing.transcode.force to re-encode anyway)"
            )
            result["skipped_reason"] = "already_target_codec"
            return result

        # Derive the container from the destination so a .webm/.mkv dest fails
        # loudly in ffmpeg rather than silently receiving MP4 bytes.
        out_suffix = os.path.splitext(dest_path)[1] or ".mp4"
        fh = tempfile.NamedTemporaryFile(delete=False, suffix=out_suffix)
        tmp_out = fh.name
        fh.close()

        cmd = [
            FFMPEG_BIN,
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-i",
            local_src,
            "-c:v",
            encoder,
            "-crf",
            str(config.get("crf", 31)),
            "-b:v",
            "0",  # required for libvpx-vp9 constant-quality (CRF) mode
            "-cpu-used",
            str(config.get("speed", 4)),
            # libvpx-vp9 is single-threaded by default; row-mt keeps this off the
            # critical path for HD clips.
            "-row-mt",
            "1",
            "-threads",
            str(_encode_threads()),
            "-pix_fmt",
            config.get("pix_fmt", "yuv420p"),
            "-an",  # augmentation outputs are silent; drop any audio stream
            tmp_out,
        ]
        logger.info(f"Transcoding output {source_codec or 'unknown'} -> {codec}...")
        proc = subprocess.run(  # noqa: S603 - intentional ffmpeg call, shell=False
            cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            hint = decode_failure_hint(stderr, local_src)
            if hint:
                logger.error(hint)
            raise RuntimeError(f"ffmpeg exited {proc.returncode}: {stderr[:500]}")

        # Never overwrite a good generation with an unverified file: ffmpeg can
        # exit 0 having written a truncated/unplayable container (disk full, or
        # a timeout kill racing the final flush).
        if os.path.getsize(tmp_out) <= 0:
            raise RuntimeError("ffmpeg produced an empty file")
        # probe_video_format returns {} for "cannot tell" (ffprobe missing, or it
        # timed out) -- that is not evidence the encode failed, and treating it as
        # such discards a good file. Only reject a codec we positively read and
        # that disagrees with the target.
        written_codec = (probe_video_format(tmp_out).get("codec_name") or "").lower()
        if written_codec and written_codec != target_stream_codec:
            raise RuntimeError(
                f"re-encoded file is {written_codec}, expected {target_stream_codec}"
            )

        # Copying into dest_path truncates it, and dest_path is usually the
        # generated video itself -- so stash a copy first and restore it if the
        # write dies midway. Otherwise a failed copy destroys the generation.
        if _dest_exists(dest_path):
            fh = tempfile.NamedTemporaryFile(delete=False, suffix=out_suffix)
            backup = fh.name
            fh.close()
            with msc.open(dest_path, "rb") as s, open(backup, "wb") as d:
                shutil.copyfileobj(s, d)

        try:
            with open(tmp_out, "rb") as s, msc.open(dest_path, "wb") as d:
                shutil.copyfileobj(s, d)
        except Exception:
            if backup:
                logger.error(f"Write to {dest_path} failed; restoring original")
                try:
                    with open(backup, "rb") as s, msc.open(dest_path, "wb") as d:
                        shutil.copyfileobj(s, d)
                except Exception as restore_err:
                    # The destination is now truncated AND the restore failed:
                    # keep the backup on disk and say where it is, rather than
                    # deleting the last intact copy in ``finally``.
                    kept, backup = backup, None
                    logger.error(
                        f"Restore of {dest_path} ALSO failed ({restore_err}). "
                        f"The only intact copy of the generated video is {kept}"
                    )
                    result["recovery_copy"] = kept
            raise

        result["transcoded"] = True
        result["output_bytes"] = os.path.getsize(tmp_out)
        logger.info(f"Transcode complete: {dest_path} is now {codec}")
        return result

    except subprocess.TimeoutExpired:
        logger.error(
            f"Transcode timed out after {_FFMPEG_TIMEOUT_S}s, keeping original output"
        )
        result["error"] = f"timeout after {_FFMPEG_TIMEOUT_S}s"
        return result
    except Exception as e:
        logger.error(f"Transcode failed, keeping original output: {e}")
        result["error"] = str(e)
        return result

    finally:
        for p in (tmp_in, tmp_out, backup):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
