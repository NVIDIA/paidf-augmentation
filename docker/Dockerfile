# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

# Use bash with pipefail so RUN steps fail on upstream errors inside pipes.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Run as non-root user for security (10000 avoids conflict with base image's UID 1000)
ARG APP_UID=10000
ARG APP_GID=10000
RUN groupadd --gid ${APP_GID} nvidia \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /bin/bash nvidia

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_MODULE_LOADING=LAZY \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    UV_LINK_MODE=copy

# -------- OS deps (build toolchain + runtime libs) --------
# FFmpeg is built from source (see below). We deliberately omit the `ffmpeg`
# binary so /usr/local/bin/ffmpeg (LGPL source build) stays first in PATH.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        jq \
        ninja-build \
        wget \
        unzip \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libwebp-dev \
        zlib1g-dev \
        pkg-config \
        yasm \
        nasm \
        libgnutls28-dev \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# -------- FFmpeg source build (LGPL, OSRB-cleared codec allow-list) --------
# Apache 2.0 posture: no GPL/nonfree. Upstream source tarballs preserved
# under /usr/share/oss-sources/source-builds/ for OSS compliance.
ARG FFMPEG_VERSION=8.1
ARG FFMPEG_SHA256=b072aed6871998cce9b36e7774033105ca29e33632be5b6347f3206898e0756a
ARG NV_CODEC_HEADERS_VERSION=n12.1.14.0
RUN set -euxo pipefail; \
    mkdir -p /tmp/ffmpeg-build /usr/share/oss-sources/source-builds && cd /tmp/ffmpeg-build; \
    git clone --depth 1 --branch "${NV_CODEC_HEADERS_VERSION}" https://github.com/FFmpeg/nv-codec-headers.git; \
    tar -C /tmp/ffmpeg-build --exclude='.git' \
        -czf "/usr/share/oss-sources/source-builds/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}.tar.gz" nv-codec-headers; \
    make -C nv-codec-headers -j"$(nproc)" && make -C nv-codec-headers install; \
    curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o ffmpeg.tar.xz; \
    echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c -; \
    cp ffmpeg.tar.xz "/usr/share/oss-sources/source-builds/ffmpeg-${FFMPEG_VERSION}.tar.xz"; \
    tar -xf ffmpeg.tar.xz; \
    cd "ffmpeg-${FFMPEG_VERSION}"; \
    ./configure \
      `# Build setup` \
      --prefix=/usr/local \
      --disable-debug --disable-doc --disable-static \
      --enable-shared --enable-pic --enable-gnutls \
      `# LGPL guard` \
      --disable-nonfree --disable-gpl --disable-vulkan --disable-cuda-nvcc \
      `# OSRB allow-list (2026-05-23): H.264 is HW-only; SW h264 stays off` \
      `# because MPEG-LA AVC covers decode as well as encode.` \
      --disable-encoders --disable-decoders --disable-everything \
      --enable-encoder=h264_nvenc --enable-encoder=mpeg4 --enable-encoder=mjpeg \
      --enable-decoder=h264_cuvid --enable-decoder=mpeg4 --enable-decoder=mjpeg \
      --enable-nvenc --enable-nvdec \
      `# Cosmos read/write IO scaffolding` \
      --enable-protocol=file \
      --enable-demuxer=mov \
      --enable-muxer=mp4 --enable-muxer=image2 \
      --enable-parser=h264 --enable-bsf=h264_mp4toannexb \
      --enable-filter=fps --enable-filter=scale --enable-filter=format \
      `# imageio_ffmpeg encoder probe: ffmpeg -f lavfi -i nullsrc -vcodec X -f null -` \
      --enable-indev=lavfi --enable-filter=nullsrc \
      --enable-decoder=wrapped_avframe --enable-muxer=null --enable-protocol=pipe \
      `# imageio_ffmpeg write path: -f rawvideo -i - (stdin via fd: in FFmpeg 8)` \
      --enable-demuxer=rawvideo --enable-decoder=rawvideo --enable-protocol=fd \
      --extra-cflags="-I/usr/local/include" \
      --extra-ldflags="-L/usr/local/lib"; \
    make -j"$(nproc)"; \
    make install; \
    ldconfig; \
    rm -rf /tmp/ffmpeg-build; \
    ffmpeg -hide_banner -version | head -n 3; \
    ! ffmpeg -buildconf | grep -E "enable-gpl|enable-nonfree"; \
    # Assert every --enable-X flag actually produced a working component. \
    # Format: <section>:<name>  (section is the ffmpeg listing flag, minus the trailing 's'). \
    for c in encoders:h264_nvenc encoders:mpeg4 encoders:mjpeg \
             decoders:h264_cuvid decoders:mpeg4 decoders:mjpeg \
             decoders:wrapped_avframe decoders:rawvideo \
             demuxers:mov demuxers:rawvideo \
             muxers:mp4 muxers:image2 muxers:null \
             protocols:file protocols:pipe protocols:fd \
             devices:lavfi bsfs:h264_mp4toannexb \
             filters:fps filters:scale filters:format filters:nullsrc; do \
        sec="${c%:*}"; name="${c#*:}"; \
        ffmpeg -hide_banner "-${sec}" 2>/dev/null | grep -qE "\\b${name}\\b" \
            || { echo "MISSING: ${name} in ${sec}"; exit 1; }; \
    done; \
    # End-to-end smoke test: lavfi -> wrapped_avframe -> mpeg4 -> null muxer via fd. \
    ffmpeg -hide_banner -f lavfi -i nullsrc=s=256x256:d=8 -vcodec mpeg4 -f null - \
        || { echo "ERROR: encoder-probe smoke test failed"; exit 1; }

RUN apt-get purge -y --auto-remove yasm nasm && rm -rf /var/lib/apt/lists/*

# Pick the yq binary + checksum for the target arch.
ARG TARGETARCH
ENV YQ_VERSION=v4.52.2
ENV YQ_BINARY=yq_linux_${TARGETARCH}
ARG YQ_SHA256_amd64=6acbc0ee0ce7d457c64407acc44a9607fff065cc436433407e27e7370c51a694
ARG YQ_SHA256_arm64=0cb66b6b49512ee1037487f139d0c531cfc34d532da365d417ccc21b4a715787
RUN mkdir -p /usr/share/oss-sources && \
    case "${TARGETARCH}" in \
        amd64) YQ_SHA256="${YQ_SHA256_amd64}" ;; \
        arm64) YQ_SHA256="${YQ_SHA256_arm64}" ;; \
        *) echo "Unsupported TARGETARCH for yq: '${TARGETARCH}'" >&2; exit 1 ;; \
    esac && \
    wget -qO /tmp/yq.tar.gz "https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/${YQ_BINARY}.tar.gz" && \
    echo "${YQ_SHA256}  /tmp/yq.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/yq.tar.gz && rm /tmp/yq.tar.gz && \
    mv ${YQ_BINARY} /usr/local/bin/yq && \
    curl -fsSL "https://github.com/mikefarah/yq/archive/refs/tags/${YQ_VERSION}.tar.gz" \
        -o "/usr/share/oss-sources/yq-${YQ_VERSION}.tar.gz"

# Install uv system-wide. Pinned to 0.11.10 (first release whose bundled
# rustls-webpki >= 0.103.13). astral.sh/uv/<ver>/install.sh redirects to
# releases.astral.sh, which BuildKit's ADD does not follow — use the official
# uv multistage image instead (the documented pinning pattern).
COPY --from=ghcr.io/astral-sh/uv:0.11.10 /uv /uvx /usr/local/bin/
ENV PATH="/usr/local/bin:$PATH"
ENV UV_HOME=/opt/uv
ARG UV_VERSION=0.11.10
RUN mkdir -p /opt/uv /usr/share/oss-sources && \
    curl -fsSL "https://github.com/astral-sh/uv/archive/refs/tags/${UV_VERSION}.tar.gz" \
        -o "/usr/share/oss-sources/uv-${UV_VERSION}.tar.gz"

# Install Python 3.13 (cosmos-transfer2.5 main + cosmos-predict2.5 v1.5.2 both
# declare .python-version=3.13 and route 3.13 builds through cosmos-dependencies/v1.5.0).
ARG PYTHON_VERSION=3.13.13
RUN uv python install ${PYTHON_VERSION} && \
    mkdir -p /usr/share/oss-sources && \
    curl -fsSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" \
        -o "/usr/share/oss-sources/Python-${PYTHON_VERSION}.tar.xz"

# Upgrade setuptools in uv's Python; `uv python find` avoids a hardcoded arch path.
RUN PYBIN="$(uv python find ${PYTHON_VERSION})" && \
    "${PYBIN}" -m pip install --no-cache-dir --break-system-packages --upgrade "setuptools>=80.9.0"

# Clone cosmos-transfer2.5 from GitHub. Tracking `main` until the next release
# tag is cut; switch back to a `vX.Y.Z` tag for reproducibility once available.
ARG COSMOS_VERSION=v1.5.4
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch ${COSMOS_VERSION} \
    https://github.com/nvidia-cosmos/cosmos-transfer2.5.git /opt/cosmos-transfer2.5

# Clone cosmos-predict2.5 from GitHub (pinned to stable release)
ARG COSMOS_PREDICT_VERSION=v1.5.2
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch ${COSMOS_PREDICT_VERSION} \
    https://github.com/nvidia-cosmos/cosmos-predict2.5.git /opt/cosmos-predict2.5

# Shared checkpoints (VAE, Qwen tokenizer, Reason1) are registered via
# cosmos_oss.checkpoints into transfer2's checkpoint_db. Predict2 uses its own
# checkpoint_db, so shared lookups fail at download time. Patch predict2's
# download_checkpoint to fall back to transfer2's registry.
RUN PYBIN="$(uv python find ${PYTHON_VERSION})" && "${PYBIN}" -c "\
import pathlib; \
p = pathlib.Path('/opt/cosmos-predict2.5/cosmos_predict2/_src/imaginaire/utils/checkpoint_db.py'); \
t = p.read_text(); \
old = '    if (checkpoint := CheckpointConfig.maybe_from_uri(checkpoint_uri)) is not None:\n        return checkpoint.download()'; \
new = old + '''\n    try:\n        from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import CheckpointConfig as _T2CC\n        if (checkpoint := _T2CC.maybe_from_uri(checkpoint_uri)) is not None:\n            return checkpoint.download()\n    except ImportError:\n        pass'''; \
patched = t.replace(old, new, 1); \
assert patched != t, 'Patch target not found in checkpoint_db.py - upstream may have changed'; \
p.write_text(patched)"

WORKDIR /app

# Copy project and generate lock so nvidia user can run venv/sync
COPY pyproject.toml ./
RUN uv lock
COPY modules /app/modules
COPY configs /app/configs

# Chown so nvidia can create venv and read Python/interpreter
RUN chown -R nvidia:nvidia /app /opt/uv /opt/cosmos-transfer2.5 /opt/cosmos-predict2.5

USER nvidia

# Create venv and install as nvidia so .venv and interpreter paths are nvidia-owned
RUN uv venv --python ${PYTHON_VERSION}

# Replicate cosmos-transfer2.5/cosmos-predict2.5's [tool.uv] override-dependencies.
# torch==2.9.1+cu130 pins nvidia-cudnn-cu13==9.13.0.50 while cosmos-oss[cu130_torch29]
# pins 9.21.0.82; unresolvable without loosening torch's pin. Upstream's
# override-dependencies setting is only honored when cosmos is the uv workspace
# root, and `uv pip install -e` from /app ignores it — so we pass it explicitly.
RUN echo 'nvidia-cudnn-cu13>=9.15.0' > /tmp/uv-overrides.txt

# arm64: glcontext has no aarch64 wheel, so it builds from source and needs X11/GL/EGL headers.
USER root
RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            libx11-dev \
            libgl1-mesa-dev \
            libegl-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi
USER nvidia

RUN uv pip install --no-cache \
    --link-mode=copy \
    --override /tmp/uv-overrides.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0 \
    --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.2.0/cu130_torch29/simple \
    --index-strategy unsafe-best-match \
    -e "/opt/cosmos-transfer2.5[cu130]"
RUN uv pip install --no-cache \
    --link-mode=copy \
    --override /tmp/uv-overrides.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0 \
    --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.2.0/cu130_torch29/simple \
    --index-strategy unsafe-best-match \
    -e "/opt/cosmos-predict2.5[cu130]"

# Rename predict2's cosmos_oss so Python falls through to transfer2's cosmos_oss.
# Then copy predict2's checkpoints_predict2.py (with HuggingFace URIs) into
# transfer2's cosmos_oss, replacing transfer2's version (which has internal S3 URIs).
RUN mv /opt/cosmos-predict2.5/packages/cosmos-oss/cosmos_oss \
       /opt/cosmos-predict2.5/packages/cosmos-oss/cosmos_oss_predict2 && \
    echo '/opt/cosmos-transfer2.5/packages/cosmos-oss' \
        > /app/.venv/lib/python3.13/site-packages/_cosmos_oss.pth
RUN cp /opt/cosmos-predict2.5/packages/cosmos-oss/cosmos_oss_predict2/checkpoints_predict2.py \
       /opt/cosmos-transfer2.5/packages/cosmos-oss/cosmos_oss/checkpoints_predict2.py

# PyTorch 2.9.0 unconditionally disables cuDNN for conv3d (vol2col fallback
# uses ~17x more memory). 2.9.1 re-enables cuDNN conv3d when runtime
# cuDNN >= 9.15 is detected (pytorch/pytorch#166480, #166643).
RUN uv pip install --no-cache --link-mode=copy \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    "torch==2.9.1"
RUN uv pip install --no-cache --link-mode=copy "nvidia-cudnn-cu13>=9.15"
RUN uv pip install --no-cache --link-mode=copy "iopath>=0.1.10"

RUN uv sync --locked --inexact --no-cache --python ${PYTHON_VERSION} --link-mode=copy

# diffusers 0.38.0 requires safetensors>=0.8.0rc0 (only released as pre-release
# at time of pin); --prerelease=allow lets uv pick it up. Scope is this single
# `uv pip install` invocation.
RUN uv pip install --no-cache --link-mode=copy --prerelease=allow \
    "nltk>=3.9.3" \
    "transformers>=4.53.0,<5.0" \
    "diffusers==0.38.0" \
    "setuptools>=80.9.0" \
    "wheel>=0.46.2" \
    "pytz>=2024.1"
# pytz: cosmos_transfer2._src.predict2.callbacks.heart_beat imports pytz
# directly. It used to come in transitively via pandas on Python 3.10; pandas
# 2.x dropped the pytz dep on Python 3.13 (zoneinfo is in stdlib), so pin
# explicitly.

# cupy-cuda13x: required for data_processing.alignment MI search (GPU-only).
# Matches the cuda:13 base. Pinned for OSRB scanning and reproducibility.
RUN uv pip install --no-cache --link-mode=copy "cupy-cuda13x==14.0.1"

# -------- OpenCV: rebuild from sdist with WITH_FFMPEG=OFF (Apache 2.0 release) --------
# PyPI opencv-python(-headless) wheels ship their own bundled FFmpeg .so files,
# which the scanner detects independently of our /usr/local FFmpeg. Force a
# CMake source build with WITH_FFMPEG=OFF; video I/O lives in PyAV (next step).
# Pinned to match pseudo-labeling for known-good scan posture.
ARG OPENCV_PYTHON_VERSION=4.12.0.88
RUN set -euxo pipefail; \
    uv pip uninstall opencv-python opencv-python-headless \
        opencv-contrib-python opencv-contrib-python-headless 2>/dev/null || true; \
    rm -rf /app/.venv/lib/python3.13/site-packages/cv2 \
           /app/.venv/lib/python3.13/site-packages/opencv_python.libs \
           /app/.venv/lib/python3.13/site-packages/opencv_python-*.dist-info \
           /app/.venv/lib/python3.13/site-packages/opencv_python_headless.libs \
           /app/.venv/lib/python3.13/site-packages/opencv_python_headless-*.dist-info \
           /app/.venv/lib/python3.13/site-packages/opencv_contrib_python*.dist-info \
           /app/.venv/lib/python3.13/site-packages/opencv_contrib_python*.libs; \
    uv pip install --no-cache --link-mode=copy \
        "setuptools>=80.9.0" "wheel>=0.46.2" \
        "scikit-build>=0.18,<1" "cmake>=3.27,<4" "numpy>=2.1.0,<3" "pip>=25.3"
RUN set -euxo pipefail; \
    ENABLE_CONTRIB=0 \
    ENABLE_HEADLESS=1 \
    CMAKE_ARGS="-DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF -DWITH_LIBV4L=OFF -DWITH_QT=OFF -DWITH_GTK=OFF -DOPENCV_ENABLE_NONFREE=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF" \
    /app/.venv/bin/python -m pip install --no-cache-dir --no-build-isolation \
        --no-binary=opencv-python \
        "opencv-python==${OPENCV_PYTHON_VERSION}"
RUN /app/.venv/bin/python -c "import re, sys, cv2; info = cv2.getBuildInformation(); sys.exit('OpenCV unexpectedly built WITH FFmpeg') if re.search(r'FFMPEG:\\s+YES', info) else print('cv2', cv2.__version__, 'WITH_FFMPEG=OFF OK')" && \
    ! find /app/.venv/lib/python3.13/site-packages -name 'cv2*.so' -exec ldd {} \; \
        | grep -E 'libav(codec|device|filter|format|util)|libswresample|libswscale'

# -------- PyAV: source build linked to /usr/local LGPL FFmpeg 8.1 --------
# PyPI `av` wheels bundle their own FFmpeg .so files (often GPL). Build from
# the upstream git tag against our /usr/local FFmpeg so the resulting av.so
# links the LGPL build we control. Version pinned to match pseudo-labeling.
ARG PYAV_VERSION=16.1.0
RUN set -euxo pipefail; \
    uv pip uninstall av 2>/dev/null || true; \
    rm -rf /app/.venv/lib/python3.13/site-packages/av \
           /app/.venv/lib/python3.13/site-packages/av-*.dist-info \
           /app/.venv/lib/python3.13/site-packages/av.libs; \
    uv pip install --no-cache --link-mode=copy \
        "setuptools>=80.9.0" "cython>=3.1.0,<4" "pip>=25.3"; \
    PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}" \
    /app/.venv/bin/python -m pip install --no-cache-dir --no-build-isolation \
        --no-binary av \
        "av==${PYAV_VERSION}"; \
    /app/.venv/bin/python -c "import av, av._core; print('pyav', av.__version__, 'core', av._core.__file__)"

# Defensive scan: fail if any opencv_python*.libs/libav* slipped back in via a
# later wheel install (would silently reintroduce a bundled FFmpeg).
RUN set -euxo pipefail; \
    leftovers="$(find /app/.venv/lib/python3.13/site-packages \( \
        -path '*opencv_python*.libs/libavcodec*' -o \
        -path '*opencv_python*.libs/libavdevice*' -o \
        -path '*opencv_python*.libs/libavfilter*' -o \
        -path '*opencv_python*.libs/libavformat*' -o \
        -path '*opencv_python*.libs/libavutil*' -o \
        -path '*opencv_python*.libs/libswresample*' -o \
        -path '*opencv_python*.libs/libswscale*' \
        -o -path '*av.libs/libav*' \
        \) -print 2>/dev/null || true)"; \
    if [ -n "$leftovers" ]; then echo "$leftovers"; exit 1; fi

# Remove the GPL-enabled ffmpeg bundled by imageio-ffmpeg (installed via cosmos).
# The container uses our LGPL source-built ffmpeg in /usr/local/bin instead.
RUN rm -rf /app/.venv/lib/python3.13/site-packages/imageio_ffmpeg/binaries/ffmpeg-*
ENV IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg

# Patch imageio_ffmpeg for non-NVENC GPUs (A100/H100): fall back to mpeg4
# (royalty-free, in our LGPL ffmpeg) when no H.264 encoder works, and pair
# -bufsize with -maxrate (mpeg4 rejects -bufsize alone).
RUN F=/app/.venv/lib/python3.13/site-packages/imageio_ffmpeg/_io.py \
 && sed -i -z 's|    else:\n        raise RuntimeError(\n            "No valid H.264|    if ffmpeg_test_encoder("mpeg4"):\n        return "mpeg4"\n    raise RuntimeError(\n            "No valid H.264|' "$F" \
 && sed -i -z 's|    cmd += \["-v", ffmpeg_log_level\]\n    cmd += output_params|&\n    if "-b:v" in cmd and "-bufsize" in cmd and "-maxrate" not in cmd:\n        cmd += ["-maxrate", cmd[cmd.index("-b:v") + 1]]|' "$F" \
 && grep -q 'ffmpeg_test_encoder("mpeg4")' "$F"

# -------- Stub decord (no FFmpeg 6 sonames in the image) --------
# decord 0.6.0 is C-linked to libav*.so.60 (FFmpeg 6); we ship FFmpeg 8 only.
# Cosmos imports decord at module top but never calls it during inference, so
# replace site-packages/decord with a stub that errors only if used (training
# paths, which this image does not ship).
RUN find /app/.venv/lib/python3.13/site-packages/decord -mindepth 1 -delete
COPY --chown=nvidia:nvidia docker/decord_stub.py \
     /app/.venv/lib/python3.13/site-packages/decord/__init__.py
RUN /app/.venv/bin/python -c "import decord; from decord import VideoReader, cpu; print('decord stub OK', decord.__version__)"

# Clean nvidia's uv cache before switching user
RUN uv cache clean

# -------- Harden: patch system Pythons & purge build-only packages --------
USER root

# Preserve sdists for the python packages we source-built, alongside FFmpeg/
# nv-codec-headers sources under /usr/share/oss-sources/source-builds/. The
# PyPI JSON entry carries both the sdist URL and its published SHA-256 digest;
# we verify the downloaded file against the latter.
RUN set -e; \
    mkdir -p /usr/share/oss-sources/source-builds; \
    for spec in "opencv-python:${OPENCV_PYTHON_VERSION}" "av:${PYAV_VERSION}"; do \
        name="${spec%%:*}"; ver="${spec##*:}"; \
        meta=$(curl -fsSL "https://pypi.org/pypi/${name}/${ver}/json" \
            | jq -r '.urls[] | select(.packagetype=="sdist") | [.url, .digests.sha256] | @tsv' | head -n1); \
        url="${meta%%$'\t'*}"; sha="${meta##*$'\t'}"; \
        [ -n "$url" ] && [ -n "$sha" ] || { echo "No sdist metadata for ${name}==${ver}"; exit 1; }; \
        out="/usr/share/oss-sources/source-builds/$(basename "$url")"; \
        curl -fsSL "$url" -o "$out"; \
        echo "${sha}  ${out}" | sha256sum -c -; \
    done

# Upgrade setuptools in every uv-managed Python the scanner can reach.
RUN for p in /root/.local/share/uv/python/cpython-*/bin/python3.* \
             /home/nvidia/.local/share/uv/python/cpython-*/bin/python3.*; do \
        [ -x "$p" ] && "$p" -m pip install --no-cache-dir --break-system-packages \
            --upgrade "setuptools>=80.9.0" 2>/dev/null || true; \
    done

RUN uv cache clean

# Remove build-only packages to shrink the attack surface.
RUN apt-get update \
    && apt-get purge -y --auto-remove \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        git-lfs \
        wget \
        unzip \
        gnupg \
        gnupg2 \
        libgnutls28-dev \
    && rm -rf /var/lib/apt/lists/*

USER nvidia

COPY --chown=nvidia:nvidia LICENSE-3rd-party.txt VERSION /app/
RUN cp /NGC-DL-CONTAINER-LICENSE /app/NGC-DL-CONTAINER-LICENSE

# Default entrypoint runs Python; override CMD to choose script
ENTRYPOINT ["uv", "run", "--no-sync", "/app/modules/cli.py"]
