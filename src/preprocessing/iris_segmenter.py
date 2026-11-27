"""Iris localization and segmentation via specular inpainting, adaptive contours, and IDO."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


DEFAULT_PREPROCESSING_CONFIG: dict[str, Any] = {
    "specular": {
        "threshold": 235,
        "dilate_radius": 3,
        "inpaint_radius": 4,
    },
    "pupil": {
        "min_radius": 18,
        "max_radius": 75,
        "border_margin": 40,
        "min_circularity": 0.50,
        "min_solidity": 0.70,
        "refine_xy_range": 6,
        "refine_r_range": 6,
    },
    "iris": {
        "min_radius": 65,
        "max_radius": 155,
        "min_iris_pupil_ratio": 1.5,
        "max_iris_pupil_ratio": 4.8,
        "center_offset_max": 14,
        "coarse_step": 3,
        "fine_step": 1,
        "lateral_arc_deg": 35,
    },
    "normalization": {
        "radial_res": 64,
        "angular_res": 256,
        "apply_clahe": True,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": [8, 8],
    },
    "roi": {
        "margin_scale": 1.15,
        "output_size": 224,
        "jpeg_quality": 95,
    },
    "quality": {
        "min_valid_pixel_ratio": 0.35,
    },
}


def merge_preprocessing_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a full preprocessing config dict merged with defaults."""
    merged = deepcopy(DEFAULT_PREPROCESSING_CONFIG)
    if not cfg:
        return merged
    for section, value in cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(section), dict):
            merged[section].update(value)
        else:
            merged[section] = deepcopy(value)
    return merged


@dataclass
class SegmentationResult:
    """Geometric parameters and diagnostic scores for a segmented eye image."""

    pupil_x: float
    pupil_y: float
    pupil_r: float
    iris_x: float
    iris_y: float
    iris_r: float
    pupil_score: float
    iris_score: float
    method: str
    success: bool


def detect_and_inpaint_specular(
    gray: np.ndarray,
    threshold: int = 235,
    dilate_radius: int = 3,
    inpaint_radius: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect specular glints and inpaint them so reflections do not distort edge detection."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    raw_mask = (gray >= threshold).astype(np.uint8) * 255
    if dilate_radius > 0:
        ksize = 2 * int(dilate_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(raw_mask, kernel)
    else:
        dilated = raw_mask

    specular_mask = dilated > 0
    if np.any(specular_mask):
        inpainted = cv2.inpaint(gray, dilated, float(inpaint_radius), cv2.INPAINT_TELEA)
    else:
        inpainted = gray.copy()

    return inpainted, specular_mask


def refine_circle_ido(
    blurred_f32: np.ndarray,
    cx: float,
    cy: float,
    r_min: int,
    r_max: int,
    xy_range: int,
    angles_rad: np.ndarray,
    step: int = 1,
) -> tuple[float, float, float, float]:
    """Vectorized Daugman Integro-Differential Operator searching a local (x, y, r) grid."""
    h, w = blurred_f32.shape
    r_min = max(2, int(r_min))
    r_max = max(r_min + 1, int(r_max))
    step = max(1, int(step))
    xy_range = max(0, int(xy_range))

    radii = np.arange(r_min - 1, r_max + 2, dtype=np.float32)
    cos_a = np.cos(angles_rad).astype(np.float32)
    sin_a = np.sin(angles_rad).astype(np.float32)

    dx = radii[:, None] * cos_a[None, :]
    dy = radii[:, None] * sin_a[None, :]

    cx_i = int(round(cx))
    cy_i = int(round(cy))
    x_start = max(0, cx_i - xy_range)
    x_end = min(w - 1, cx_i + xy_range)
    y_start = max(0, cy_i - xy_range)
    y_end = min(h - 1, cy_i + xy_range)

    min_valid_angles = max(3, int(0.6 * len(angles_rad)))

    best_score = -np.inf
    best_x = float(cx)
    best_y = float(cy)
    best_r = float(0.5 * (r_min + r_max))

    for y in range(y_start, y_end + 1, step):
        ys = np.rint(y + dy).astype(np.int32)
        y_ok = (ys >= 0) & (ys < h)
        ys_clip = np.clip(ys, 0, h - 1)

        for x in range(x_start, x_end + 1, step):
            xs = np.rint(x + dx).astype(np.int32)
            valid = y_ok & (xs >= 0) & (xs < w)
            valid_counts = valid.sum(axis=1)

            if np.max(valid_counts) < min_valid_angles:
                continue

            xs_clip = np.clip(xs, 0, w - 1)
            vals = blurred_f32[ys_clip, xs_clip]
            profile = (vals * valid).sum(axis=1) / np.maximum(valid_counts, 1)

            padded = np.pad(profile, 1, mode="edge")
            smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
            deriv = (smoothed[2:] - smoothed[:-2]) * 0.5

            valid_r = (
                (valid_counts[1:-1] >= min_valid_angles)
                & (valid_counts[:-2] >= min_valid_angles)
                & (valid_counts[2:] >= min_valid_angles)
            )
            if not np.any(valid_r):
                continue

            deriv = np.where(valid_r, deriv, -np.inf)
            max_idx = int(np.argmax(deriv))
            score = float(deriv[max_idx])
            if score > best_score:
                best_score = score
                best_x = float(x)
                best_y = float(y)
                best_r = float(r_min + max_idx)

    if not np.isfinite(best_score):
        return float(cx), float(cy), float(0.5 * (r_min + r_max)), 0.0

    return best_x, best_y, best_r, best_score


def _mean_circle_intensity(gray: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """Compute mean pixel intensity inside a circle of given center and radius."""
    h, w = gray.shape
    r_inner = max(2, int(round(0.7 * radius)))
    x0 = max(0, int(round(cx)) - r_inner)
    x1 = min(w, int(round(cx)) + r_inner + 1)
    y0 = max(0, int(round(cy)) - r_inner)
    y1 = min(h, int(round(cy)) + r_inner + 1)
    if x1 <= x0 or y1 <= y0:
        return 255.0

    patch = gray[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_inner ** 2
    if not np.any(mask):
        return 255.0
    return float(patch[mask].mean())


def detect_pupil(
    inpainted: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[float, float, float, float, str]:
    """Detect pupil circle using adaptive dark contours + IDO, with HoughCircles fallback."""
    p_cfg = cfg["pupil"]
    min_r = int(p_cfg["min_radius"])
    max_r = int(p_cfg["max_radius"])
    border = int(p_cfg["border_margin"])
    min_circ = float(p_cfg["min_circularity"])
    min_sol = float(p_cfg["min_solidity"])
    xy_range = int(p_cfg["refine_xy_range"])
    r_range = int(p_cfg["refine_r_range"])

    h, w = inpainted.shape
    blurred_u8 = cv2.GaussianBlur(inpainted, (5, 5), 0)
    blurred_f32 = blurred_u8.astype(np.float32)
    full_angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)

    # Central patch reduces bias from dark borders or eyeglass frames
    y_lo, y_hi = int(0.15 * h), int(0.85 * h)
    x_lo, x_hi = int(0.15 * w), int(0.85 * w)
    center_patch = blurred_u8[y_lo:y_hi, x_lo:x_hi]
    pct_thresholds = [int(round(float(np.percentile(center_patch, p)))) for p in (3, 6, 10)]
    thresholds = sorted({max(15, min(130, t)) for t in ([35, 45, 55, 65] + pct_thresholds)})

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    min_area = np.pi * (min_r ** 2) * 0.6
    max_area = np.pi * (max_r ** 2) * 1.4
    diag_half = float(np.hypot(w / 2.0, h / 2.0))

    candidates: list[tuple[float, float, float, float]] = []
    for thresh in thresholds:
        binary = (blurred_u8 <= thresh).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perim = cv2.arcLength(cnt, True)
            if perim <= 1e-6:
                continue
            circularity = float(4.0 * np.pi * area / (perim * perim))
            if circularity < min_circ:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 1e-6:
                continue
            solidity = float(area / hull_area)
            if solidity < min_sol:
                continue

            moments = cv2.moments(cnt)
            if moments["m00"] <= 1e-6:
                continue
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            rad = float(np.sqrt(area / np.pi))

            if rad < min_r or rad > max_r:
                continue
            if cx < border or cx > w - border or cy < border or cy > h - border:
                continue

            mean_inside = _mean_circle_intensity(blurred_u8, cx, cy, rad)
            dist_norm = float(np.hypot(cx - w / 2.0, cy - h / 2.0)) / max(diag_half, 1.0)
            cand_score = (255.0 - mean_inside) * circularity * solidity - 15.0 * dist_norm
            candidates.append((cand_score, cx, cy, rad))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        deduped: list[tuple[float, float, float]] = []
        for _, cx, cy, rad in candidates:
            if all(np.hypot(cx - kx, cy - ky) >= 15.0 for kx, ky, _ in deduped):
                deduped.append((cx, cy, rad))
            if len(deduped) >= 3:
                break

        best_ido: tuple[float, float, float, float] | None = None
        for cx, cy, rad in deduped:
            rx, ry, rr, rscore = refine_circle_ido(
                blurred_f32,
                cx,
                cy,
                max(min_r, int(round(rad)) - r_range),
                min(max_r, int(round(rad)) + r_range),
                xy_range=xy_range,
                angles_rad=full_angles,
                step=1,
            )
            if best_ido is None or rscore > best_ido[3]:
                best_ido = (rx, ry, rr, rscore)

        if best_ido is not None:
            return best_ido[0], best_ido[1], best_ido[2], best_ido[3], "contour_ido"

    # Stage 2: HoughCircles fallback inside central region
    use_crop = (w > 2 * border) and (h > 2 * border)
    off_x = border if use_crop else 0
    off_y = border if use_crop else 0
    crop = blurred_u8[off_y : h - off_y, off_x : w - off_x] if use_crop else blurred_u8

    circles = cv2.HoughCircles(
        crop,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=25,
        param1=80,
        param2=22,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is not None and len(circles[0]) > 0:
        hough_cands: list[tuple[float, float, float, float]] = []
        for cx_c, cy_c, rad_c in circles[0][:5]:
            cx = float(cx_c) + off_x
            cy = float(cy_c) + off_y
            rad = float(np.clip(rad_c, min_r, max_r))
            darkness = 255.0 - _mean_circle_intensity(blurred_u8, cx, cy, rad)
            hough_cands.append((darkness, cx, cy, rad))

        hough_cands.sort(key=lambda item: item[0], reverse=True)
        best_ido = None
        for _, cx, cy, rad in hough_cands[:3]:
            rx, ry, rr, rscore = refine_circle_ido(
                blurred_f32,
                cx,
                cy,
                max(min_r, int(round(rad)) - r_range),
                min(max_r, int(round(rad)) + r_range),
                xy_range=xy_range,
                angles_rad=full_angles,
                step=1,
            )
            if best_ido is None or rscore > best_ido[3]:
                best_ido = (rx, ry, rr, rscore)

        if best_ido is not None:
            return best_ido[0], best_ido[1], best_ido[2], best_ido[3], "hough_ido"

    # Stage 3: Safe image-center fallback
    return float(w / 2.0), float(h / 2.0), 35.0, 0.0, "fallback"


def detect_iris(
    inpainted: np.ndarray,
    px: float,
    py: float,
    pr: float,
    cfg: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Detect outer iris/sclera boundary using lateral arcs to avoid eyelid occlusion."""
    i_cfg = cfg["iris"]
    blurred_f32 = cv2.GaussianBlur(inpainted, (7, 7), 0).astype(np.float32)

    lat_deg = float(i_cfg["lateral_arc_deg"])
    right_arc = np.deg2rad(np.linspace(-lat_deg, lat_deg, 31))
    left_arc = np.deg2rad(np.linspace(180.0 - lat_deg, 180.0 + lat_deg, 31))
    angles_rad = np.concatenate([right_arc, left_arc])

    min_r_cfg = int(i_cfg["min_radius"])
    max_r_cfg = int(i_cfg["max_radius"])
    r_lo = max(min_r_cfg, int(round(pr * float(i_cfg["min_iris_pupil_ratio"]))))
    r_hi = min(max_r_cfg, int(round(pr * float(i_cfg["max_iris_pupil_ratio"]))))

    if r_lo + 4 > r_hi:
        r_hi = max(r_lo + 4, max_r_cfg)
        if r_lo + 4 > r_hi:
            r_lo = max(4, r_hi - 4)

    coarse_step = int(i_cfg["coarse_step"])
    fine_step = int(i_cfg["fine_step"])
    offset_max = int(i_cfg["center_offset_max"])

    ix0, iy0, ir0, _ = refine_circle_ido(
        blurred_f32,
        px,
        py,
        r_lo,
        r_hi,
        xy_range=offset_max,
        angles_rad=angles_rad,
        step=coarse_step,
    )

    fine_r_lo = max(r_lo, int(round(ir0)) - 4)
    fine_r_hi = min(r_hi, int(round(ir0)) + 4)
    if fine_r_lo + 1 > fine_r_hi:
        fine_r_hi = fine_r_lo + 1

    return refine_circle_ido(
        blurred_f32,
        ix0,
        iy0,
        fine_r_lo,
        fine_r_hi,
        xy_range=coarse_step,
        angles_rad=angles_rad,
        step=fine_step,
    )


def segment_iris(
    gray: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> tuple[SegmentationResult, np.ndarray, np.ndarray]:
    """Run specular removal, pupil localization, and iris localization on a grayscale image."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    merged_cfg = merge_preprocessing_config(cfg)
    spec_cfg = merged_cfg["specular"]

    inpainted, specular_mask = detect_and_inpaint_specular(
        gray,
        threshold=int(spec_cfg["threshold"]),
        dilate_radius=int(spec_cfg["dilate_radius"]),
        inpaint_radius=int(spec_cfg["inpaint_radius"]),
    )
    px, py, pr, p_score, method = detect_pupil(inpainted, merged_cfg)
    ix, iy, ir, i_score = detect_iris(inpainted, px, py, pr, merged_cfg)

    center_dist = float(np.hypot(ix - px, iy - py))
    ratio = ir / max(pr, 1e-6)
    success = (
        method != "fallback"
        and (center_dist + pr < ir - 4.0)
        and (1.4 <= ratio <= 5.0)
        and (p_score > 3.0)
        and (i_score > 1.5)
    )

    result = SegmentationResult(
        pupil_x=float(px),
        pupil_y=float(py),
        pupil_r=float(pr),
        iris_x=float(ix),
        iris_y=float(iy),
        iris_r=float(ir),
        pupil_score=float(p_score),
        iris_score=float(i_score),
        method=method,
        success=bool(success),
    )
    return result, inpainted, specular_mask
