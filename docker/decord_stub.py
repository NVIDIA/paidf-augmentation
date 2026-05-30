# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stub decord package.

Replaced into site-packages/decord at image build time to keep the runtime
free of FFmpeg 6 sonames (libavformat.so.60 / libavcodec60 / libavutil58 /
libswresample4 / libswscale7). The real decord 0.6.0 wheel is C-linked
against those, and we ship only the LGPL source-built FFmpeg 8
(.so.62 / .60 / .11 / .9 / .6).

Cosmos modules import decord at module top but only *use* it inside
training / dataset-loading code that this distribution does not exercise.
Attempting to construct VideoReader / call decord.X() here raises
NotImplementedError with a pointer back to this comment.
"""

__version__ = "0.0.0-stub"

_MSG = (
    "decord is stubbed in this image (OSRB-clean build, no FFmpeg 6 sonames). "
    "Cosmos training / dataset paths that decode video via decord are not "
    "supported here. Rebuild against a real decord if you need them."
)


class _Stub:
    def __init__(self, name):
        self._n = name

    def __call__(self, *_a, **_k):
        raise NotImplementedError(f"decord.{self._n}(): {_MSG}")

    def __getattr__(self, n):
        raise NotImplementedError(f"decord.{self._n}.{n}: {_MSG}")


VideoReader = _Stub("VideoReader")
VideoLoader = _Stub("VideoLoader")
AudioReader = _Stub("AudioReader")
AVReader = _Stub("AVReader")
cpu = _Stub("cpu")
gpu = _Stub("gpu")
bridge = _Stub("bridge")


def __getattr__(name):
    return _Stub(name)
