# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MI-registration alignment of an augmented output back to its input frame.

Implements a 5-DOF affine (scale-X, scale-Y, rotation, tx, ty) mutual
information search on GPU (cupy required), then warps + crops the align image
into the reference frame. Reference implementation:
``data/alignment_module/_registration.py``.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

import cupy as cp
import cv2
import multistorageclient as msc
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _to_gray(img: NDArray) -> NDArray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[-1] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _downsample_2x(gray: NDArray) -> NDArray:
    h, w = gray.shape
    nh, nw = h // 2, w // 2
    return (
        gray[0 : nh * 2 : 2, 0 : nw * 2 : 2].astype(np.uint16)
        + gray[1 : nh * 2 : 2, 0 : nw * 2 : 2]
        + gray[0 : nh * 2 : 2, 1 : nw * 2 : 2]
        + gray[1 : nh * 2 : 2, 1 : nw * 2 : 2]
    ) // 4


def _build_pyramid(gray: NDArray, levels: int) -> list[NDArray]:
    pyr = [gray]
    for _ in range(1, levels):
        prev = pyr[-1]
        if prev.shape[0] < 32 or prev.shape[1] < 32:
            break
        pyr.append(_downsample_2x(prev).astype(np.uint8))
    pyr.reverse()
    return pyr


# ---------------------------------------------------------------------------
# GPU MI evaluation (cupy)
# ---------------------------------------------------------------------------


def _gpu_precompute(ref_gpu, test_gpu, num_bins: int) -> dict:
    rh, rw = ref_gpu.shape
    th, tw = test_gpu.shape
    bin_size = 256.0 / num_bins

    test_f = test_gpu.astype(cp.float32).ravel()
    ref_bins = cp.minimum(
        (ref_gpu.astype(cp.float32) / bin_size).astype(cp.int32),
        num_bins - 1,
    ).ravel()

    yy, xx = cp.mgrid[0:rh, 0:rw].astype(cp.float32)
    dx = (xx - rw / 2.0).ravel()
    dy = (yy - rh / 2.0).ravel()

    return {
        "rh": rh,
        "rw": rw,
        "th": th,
        "tw": tw,
        "n_pixels": int(rh * rw),
        "bin_size": bin_size,
        "num_bins": num_bins,
        "test_f": test_f,
        "ref_bins": ref_bins,
        "dx": dx,
        "dy": dy,
        "cx_test": tw / 2.0,
        "cy_test": th / 2.0,
    }


def _gpu_batch_eval_mi(
    gpu_ctx: dict, combos: NDArray, chunk_size: int | None = None
) -> NDArray:
    th, tw = gpu_ctx["th"], gpu_ctx["tw"]
    n_pixels = gpu_ctx["n_pixels"]
    num_bins = gpu_ctx["num_bins"]
    bin_size = gpu_ctx["bin_size"]
    test_f = gpu_ctx["test_f"]
    ref_bins = gpu_ctx["ref_bins"]
    dx, dy = gpu_ctx["dx"], gpu_ctx["dy"]
    cx_test, cy_test = gpu_ctx["cx_test"], gpu_ctx["cy_test"]

    combos_gpu = cp.asarray(combos, dtype=cp.float32)
    n_combos = int(combos_gpu.shape[0])

    if chunk_size is None:
        free, _ = cp.cuda.Device().mem_info
        per_combo = n_pixels * 4 * 12 + num_bins * num_bins * 4
        chunk_size = max(1, int(free * 0.25 // per_combo))
        chunk_size = min(chunk_size, n_combos)

    results = cp.empty(n_combos, dtype=cp.float32)
    min_count = n_pixels * 0.05

    for start in range(0, n_combos, chunk_size):
        end = min(start + chunk_size, n_combos)
        b = end - start
        batch = combos_gpu[start:end]
        sx = batch[:, 0:1]
        sy = batch[:, 1:2]
        rot = batch[:, 2:3]
        tx = batch[:, 3:4]
        ty = batch[:, 4:5]
        cos_r = cp.cos(rot)
        sin_r = cp.sin(rot)

        xt = sx * (cos_r * dx - sin_r * dy) + cx_test + tx
        yt = sy * (sin_r * dx + cos_r * dy) + cy_test + ty

        x0 = cp.floor(xt).astype(cp.int32)
        y0 = cp.floor(yt).astype(cp.int32)
        valid = (x0 >= 0) & (x0 < tw - 1) & (y0 >= 0) & (y0 < th - 1)
        fx = xt - x0
        fy = yt - y0

        x0c = cp.clip(x0, 0, tw - 2)
        y0c = cp.clip(y0, 0, th - 2)
        idx00 = y0c * tw + x0c
        val = (
            (1 - fx) * (1 - fy) * test_f[idx00]
            + fx * (1 - fy) * test_f[idx00 + 1]
            + (1 - fx) * fy * test_f[idx00 + tw]
            + fx * fy * test_f[idx00 + tw + 1]
        )

        warped_bins = cp.minimum((val / bin_size).astype(cp.int32), num_bins - 1)
        del xt, yt, x0, y0, fx, fy, x0c, y0c, idx00, val

        global_idx = (
            cp.arange(b, dtype=cp.int32).reshape(b, 1) * (num_bins * num_bins)
            + ref_bins * num_bins
            + warped_bins
        )
        weights = valid.astype(cp.float32).ravel()
        hist = cp.bincount(
            global_idx.ravel(),
            weights=weights,
            minlength=b * num_bins * num_bins,
        ).reshape(b, num_bins, num_bins)
        del global_idx, warped_bins, weights, valid

        count = hist.sum(axis=(1, 2))
        h_a = hist.sum(axis=2)
        h_b = hist.sum(axis=1)
        count_safe = cp.maximum(count, 1.0)
        pab = hist / count_safe.reshape(b, 1, 1)
        pa = h_a / count_safe.reshape(b, 1)
        pb = h_b / count_safe.reshape(b, 1)
        outer = pa.reshape(b, num_bins, 1) * pb.reshape(b, 1, num_bins)

        safe = (pab > 0) & (outer > 0)
        safe_pab = cp.where(safe, pab, 1.0)
        safe_outer = cp.where(safe, outer, 1.0)
        log_term = cp.where(safe, cp.log(safe_pab / safe_outer), 0.0)
        mi = (pab * log_term).sum(axis=(1, 2))
        mi = mi * (count / n_pixels)
        mi = cp.where(count >= min_count, mi, -cp.inf)
        results[start:end] = mi

    return cp.asnumpy(results)


def _eval_one(
    gpu_ctx: dict, sx: float, sy: float, rot: float, tx: float, ty: float
) -> float:
    combo = np.array([[sx, sy, rot, tx, ty]], dtype=np.float32)
    return float(_gpu_batch_eval_mi(gpu_ctx, combo, chunk_size=1)[0])


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------


def _grid_search_gpu(
    gpu_ctx: dict,
    sx_vals,
    sy_vals,
    rot_vals,
    tx_vals,
    ty_vals,
) -> tuple[float, float, float, float, float, float]:
    mesh = np.meshgrid(sx_vals, sy_vals, rot_vals, tx_vals, ty_vals, indexing="ij")
    combos = np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32)
    mi_vals = _gpu_batch_eval_mi(gpu_ctx, combos)
    best = int(np.argmax(mi_vals))
    bp = combos[best]
    return (
        float(mi_vals[best]),
        float(bp[0]),
        float(bp[1]),
        float(bp[2]),
        float(bp[3]),
        float(bp[4]),
    )


def _coordinate_descent(
    gpu_ctx: dict,
    sx,
    sy,
    rot_rad,
    tx,
    ty,
    step_sx,
    step_sy,
    step_r,
    step_t,
    max_iter: int = 200,
):
    best_mi = _eval_one(gpu_ctx, sx, sy, rot_rad, tx, ty)
    for _ in range(max_iter):
        improved = False
        for name, step in (
            ("sX", step_sx),
            ("sY", step_sy),
            ("rot", step_r),
            ("tx", step_t),
            ("ty", step_t),
        ):
            orig = {"sX": sx, "sY": sy, "rot": rot_rad, "tx": tx, "ty": ty}[name]
            for direction in (1, -1):
                new = orig + direction * step
                if name == "sX":
                    sx = new
                elif name == "sY":
                    sy = new
                elif name == "rot":
                    rot_rad = new
                elif name == "tx":
                    tx = new
                else:
                    ty = new
                mi = _eval_one(gpu_ctx, sx, sy, rot_rad, tx, ty)
                if mi > best_mi + 1e-8:
                    best_mi = mi
                    improved = True
                    break
                if name == "sX":
                    sx = orig
                elif name == "sY":
                    sy = orig
                elif name == "rot":
                    rot_rad = orig
                elif name == "tx":
                    tx = orig
                else:
                    ty = orig

        if not improved:
            step_sx *= 0.5
            step_sy *= 0.5
            step_r *= 0.5
            step_t *= 0.5
            if (
                step_sx < 0.001
                and step_sy < 0.001
                and step_r < 0.0002
                and step_t < 0.25
            ):
                break
    return best_mi, sx, sy, rot_rad, tx, ty


# ---------------------------------------------------------------------------
# High-level registration + warp
# ---------------------------------------------------------------------------


def _register(
    ref_gray: NDArray,
    test_gray: NDArray,
    cfg: dict,
    logger: logging.Logger,
) -> dict:
    DEG2RAD = math.pi / 180.0

    sx_min, sx_max, sx_step = cfg["sx_range"]
    sy_min, sy_max, sy_step = cfg["sy_range"]
    rot_min_deg, rot_max_deg, rot_step_deg = cfg["rot_range_deg"]
    shift_range = cfg["shift_range"]
    shift_step = cfg["shift_step"]
    num_bins = cfg["bins"]
    pyr_levels = cfg["pyr_levels"]

    ref_pyr = _build_pyramid(ref_gray, pyr_levels)
    test_pyr = _build_pyramid(test_gray, pyr_levels)
    total = len(ref_pyr)
    logger.info(
        f"Registration pyramid {total} level(s): "
        + ", ".join(f"{p.shape[1]}x{p.shape[0]}" for p in ref_pyr)
    )

    cur_sx = cur_sy = 1.0
    cur_rot = cur_tx = cur_ty = 0.0
    best_mi = -math.inf
    gpu_ctx: dict = {}

    for lvl in range(total):
        ref_lvl, test_lvl = ref_pyr[lvl], test_pyr[lvl]
        level_factor = 2 ** (total - 1 - lvl)

        gpu_ctx = _gpu_precompute(cp.asarray(ref_lvl), cp.asarray(test_lvl), num_bins)

        if lvl == 0:
            sr = math.ceil(shift_range / level_factor)
            ss = max(1, math.ceil(shift_step / level_factor * 2))
            sx_vals = np.arange(sx_min, sx_max + 0.001, sx_step * 2)
            sy_vals = np.arange(sy_min, sy_max + 0.001, sy_step * 2)
            rot_vals = (
                np.arange(rot_min_deg, rot_max_deg + 0.001, rot_step_deg) * DEG2RAD
            )
            tx_vals = np.arange(-sr, sr + 1, ss, dtype=np.float64)
            ty_vals = np.arange(-sr, sr + 1, ss, dtype=np.float64)

            best_mi, cur_sx, cur_sy, cur_rot, cur_tx, cur_ty = _grid_search_gpu(
                gpu_ctx,
                sx_vals,
                sy_vals,
                rot_vals,
                tx_vals,
                ty_vals,
            )
            logger.info(
                f"  level {lvl + 1} grid best MI={best_mi:.4f} "
                f"sX={cur_sx:.3f} sY={cur_sy:.3f} rot={math.degrees(cur_rot):.2f} "
                f"tx={cur_tx:.1f} ty={cur_ty:.1f}"
            )
        else:
            cur_tx *= 2
            cur_ty *= 2

        step_sx = sx_step if lvl == 0 else sx_step / 2
        step_sy = sy_step if lvl == 0 else sy_step / 2
        step_r = (rot_step_deg * DEG2RAD) if lvl == 0 else (rot_step_deg * DEG2RAD / 2)
        step_t = max(1, math.ceil(shift_step / level_factor)) if lvl == 0 else 2.0

        best_mi, cur_sx, cur_sy, cur_rot, cur_tx, cur_ty = _coordinate_descent(
            gpu_ctx,
            cur_sx,
            cur_sy,
            cur_rot,
            cur_tx,
            cur_ty,
            step_sx,
            step_sy,
            step_r,
            step_t,
        )

    mi_before = _eval_one(gpu_ctx, 1.0, 1.0, 0.0, 0.0, 0.0)
    return {
        "scaleX": float(cur_sx),
        "scaleY": float(cur_sy),
        "rotation_deg": float(math.degrees(cur_rot)),
        "tx": float(cur_tx),
        "ty": float(cur_ty),
        "mi_before": float(mi_before),
        "mi_after": float(best_mi),
    }


def _build_affine(
    params: dict, ref_w: int, ref_h: int, test_w: int, test_h: int
) -> NDArray:
    sx, sy = params["scaleX"], params["scaleY"]
    rot = math.radians(params["rotation_deg"])
    tx, ty = params["tx"], params["ty"]
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    cx_ref, cy_ref = ref_w / 2.0, ref_h / 2.0
    cx_test, cy_test = test_w / 2.0, test_h / 2.0
    return np.array(
        [
            [
                sx * cos_r,
                -sx * sin_r,
                -sx * cos_r * cx_ref + sx * sin_r * cy_ref + cx_test + tx,
            ],
            [
                sy * sin_r,
                sy * cos_r,
                -sy * sin_r * cx_ref - sy * cos_r * cy_ref + cy_test + ty,
            ],
        ],
        dtype=np.float64,
    )


def _warp(test_img: NDArray, params: dict, ref_shape: tuple[int, int], interp: str):
    ref_h, ref_w = ref_shape
    th, tw = test_img.shape[:2]
    M = _build_affine(params, ref_w, ref_h, tw, th)
    interp_flag = cv2.INTER_NEAREST if interp == "nearest" else cv2.INTER_LINEAR
    flags = interp_flag | cv2.WARP_INVERSE_MAP
    warped = cv2.warpAffine(test_img, M, (ref_w, ref_h), flags=flags)
    mask = cv2.warpAffine(
        np.ones((th, tw), dtype=np.uint8),
        M,
        (ref_w, ref_h),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    )
    return warped, mask


def _valid_bbox(mask: NDArray) -> tuple[int, int, int, int]:
    """Bbox (x0, y0, x1, y1) of valid (non-zero) mask pixels."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Registration mask has no valid pixels.")
    h, w = mask.shape
    x0, y0 = max(0, int(xs.min())), max(0, int(ys.min()))
    x1, y1 = min(w, int(xs.max()) + 1), min(h, int(ys.max()) + 1)
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _read_image(path: str) -> NDArray:
    with msc.open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _write_image(path: str, img: NDArray) -> None:
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {path}")
    with msc.open(path, "wb") as f:
        f.write(buf.tobytes())


def run_alignment(
    ref_path: str,
    align_path: str,
    dest_path: str,
    config: dict[str, Any],
    logger: logging.Logger,
) -> Optional[dict]:
    """Align ``align_path`` into ``ref_path``'s frame and overwrite ``dest_path``.

    Args:
        ref_path: Reference (input) image — the alignment target frame.
        align_path: Generated image to warp into the reference frame.
        dest_path: Where to write the aligned image (typically ``data.output.video``).
        config: ``data_processing.alignment`` parameters (Pydantic-validated).
        logger: pipeline logger.

    Returns:
        Dict with the recovered transform, MI before/after, and bbox; or
        ``None`` when gated by ``min_mi``.
    """
    interp = config["interp"]
    no_resize = config["no_resize"]

    logger.info(
        f"Alignment: ref={ref_path}  align={align_path}  dest={dest_path}  "
        f"interp={interp}"
    )

    ref_img = _read_image(ref_path)
    align_img = _read_image(align_path)
    ref_gray = _to_gray(ref_img)
    align_gray = _to_gray(align_img)

    rh, rw = ref_gray.shape
    if not no_resize and align_gray.shape != (rh, rw):
        align_gray = cv2.resize(align_gray, (rw, rh), interpolation=cv2.INTER_LINEAR)
        align_img = cv2.resize(align_img, (rw, rh), interpolation=cv2.INTER_LINEAR)

    # Auto-derive dimension-dependent parameters from image sizes.
    # Explicit values in the config win; leave them null to use these defaults.
    ah, aw = align_gray.shape
    if config.get("sx_range") is None:
        rx = aw / rw
        config["sx_range"] = [round(rx * 0.9, 3), round(rx * 1.1, 3), 0.1]
        logger.info(
            f"Alignment: auto sx_range={config['sx_range']} (ratio {rx:.3f} ±10%)"
        )
    if config.get("sy_range") is None:
        ry = ah / rh
        config["sy_range"] = [round(ry * 0.9, 3), round(ry * 1.1, 3), 0.1]
        logger.info(
            f"Alignment: auto sy_range={config['sy_range']} (ratio {ry:.3f} ±10%)"
        )
    if config.get("shift_range") is None:
        sr_auto = max(rh, rw) // 4
        config["shift_range"] = sr_auto
        logger.info(f"Alignment: auto shift_range={sr_auto} (max(ref)/4)")
    if config.get("pyr_levels") is None:
        max_levels, d = 1, min(rh, rw)
        while d >= 64:
            max_levels += 1
            d //= 2
        pyr_auto = min(3, max_levels)
        config["pyr_levels"] = pyr_auto
        logger.info(
            f"Alignment: auto pyr_levels={pyr_auto} "
            f"(ref {rw}x{rh} supports up to {max_levels})"
        )

    t0 = time.perf_counter()
    params = _register(ref_gray, align_gray, config, logger)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Alignment: MI {params['mi_before']:.4f} -> {params['mi_after']:.4f}  "
        f"sx={params['scaleX']:.3f} sy={params['scaleY']:.3f} "
        f"rot={params['rotation_deg']:.2f} t=({params['tx']:.1f},{params['ty']:.1f})  "
        f"({elapsed:.1f}s)"
    )

    min_mi = config.get("min_mi")
    if min_mi is not None and params["mi_after"] < float(min_mi):
        logger.error(
            f"Alignment MI {params['mi_after']:.4f} below min_mi={min_mi}; aborting."
        )
        return None

    warped, mask = _warp(align_img, params, ref_gray.shape, interp)
    bbox = _valid_bbox(mask)
    params["bbox"] = list(bbox)
    logger.info(
        f"Alignment: bbox={bbox} -> aligned {warped.shape[1]}x{warped.shape[0]} "
        f"(zero-padded outside bbox)"
    )

    _write_image(dest_path, warped)
    logger.info(f"Alignment: wrote {dest_path}")
    return params
