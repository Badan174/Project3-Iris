"""Daugman Rubber Sheet normalization, ROI cropping, and QA visualization."""

from __future__ import annotations

import cv2
import numpy as np

from .iris_segmenter import SegmentationResult


def normalize_rubber_sheet(
    gray: np.ndarray,
    seg: SegmentationResult,
    specular_mask: np.ndarray | None = None,
    radial_res: int = 64,
    angular_res: int = 256,
    apply_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: tuple[int, int] = (8, 8),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unwrap annular iris texture into a fixed polar rectangle via ray-circle intersection."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    theta = np.linspace(0.0, 2.0 * np.pi, angular_res, endpoint=False, dtype=np.float32)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Pupil boundary point along each ray from (pupil_x, pupil_y)
    xp = seg.pupil_x + seg.pupil_r * cos_t
    yp = seg.pupil_y + seg.pupil_r * sin_t

    # Exact ray intersection from pupil center with non-concentric outer iris circle
    dx_pi = float(seg.pupil_x - seg.iris_x)
    dy_pi = float(seg.pupil_y - seg.iris_y)
    b = dx_pi * cos_t + dy_pi * sin_t
    c = dx_pi * dx_pi + dy_pi * dy_pi - float(seg.iris_r) ** 2
    disc = np.maximum(b * b - c, 0.0)
    dist = np.maximum(-b + np.sqrt(disc), float(seg.pupil_r) + 1.0)

    xi = seg.pupil_x + dist * cos_t
    yi = seg.pupil_y + dist * sin_t

    # Sample strictly inside (0, 1) to stay clear of pupil and sclera transition pixels
    r = (np.arange(radial_res, dtype=np.float32) + 0.5) / float(radial_res)
    map_x = ((1.0 - r[:, None]) * xp[None, :] + r[:, None] * xi[None, :]).astype(np.float32)
    map_y = ((1.0 - r[:, None]) * yp[None, :] + r[:, None] * yi[None, :]).astype(np.float32)

    polar_raw = cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    in_bounds = (map_x >= 0.0) & (map_x <= float(w - 1)) & (map_y >= 0.0) & (map_y <= float(h - 1))
    if specular_mask is not None:
        spec_u8 = specular_mask.astype(np.uint8) * 255
        spec_polar = (
            cv2.remap(
                spec_u8,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            > 0
        )
    else:
        spec_polar = np.zeros_like(in_bounds, dtype=bool)

    base_valid = in_bounds & (~spec_polar)
    angles_deg = np.rad2deg(theta) % 360.0
    lateral_cols = (
        (angles_deg <= 35.0)
        | (angles_deg >= 325.0)
        | ((angles_deg >= 145.0) & (angles_deg <= 215.0))
    )

    lat_valid = base_valid[:, lateral_cols]
    if np.any(lat_valid):
        ref_pixels = polar_raw[:, lateral_cols][lat_valid].astype(np.float32)
    elif np.any(base_valid):
        ref_pixels = polar_raw[base_valid].astype(np.float32)
    else:
        ref_pixels = polar_raw.astype(np.float32).ravel()

    med = float(np.median(ref_pixels))
    mad = float(np.median(np.abs(ref_pixels - med)))
    sigma_rob = max(1.4826 * mad, 15.0)

    low_thresh = max(15.0, med - 3.0 * sigma_rob)
    high_thresh = min(240.0, med + 3.0 * sigma_rob)

    valid_bool = base_valid & (polar_raw >= low_thresh) & (polar_raw <= high_thresh)
    valid_mask = valid_bool.astype(np.uint8) * 255

    if apply_clahe:
        # Fill occluded regions with median iris intensity before CLAHE to avoid halo artifacts
        filled = polar_raw.copy()
        filled[~valid_bool] = np.uint8(np.clip(round(med), 0, 255))
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(int(clahe_tile_grid[0]), int(clahe_tile_grid[1])),
        )
        polar_enhanced = clahe.apply(filled)
    else:
        polar_enhanced = polar_raw.copy()

    return polar_raw, polar_enhanced, valid_mask


def extract_iris_roi(
    gray: np.ndarray,
    seg: SegmentationResult,
    margin_scale: float = 1.15,
    output_size: int = 224,
) -> np.ndarray:
    """Crop a square region centered on the iris with reflection padding near borders."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    cx = int(round(seg.iris_x))
    cy = int(round(seg.iris_y))
    half = int(round(max(seg.iris_r * margin_scale, 40.0)))

    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded = cv2.copyMakeBorder(
            gray,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        crop = padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
    else:
        crop = gray[y1:y2, x1:x2]

    interp = cv2.INTER_AREA if crop.shape[0] > output_size else cv2.INTER_LINEAR
    return cv2.resize(crop, (int(output_size), int(output_size)), interpolation=interp)


def draw_segmentation_overlay(gray: np.ndarray, seg: SegmentationResult) -> np.ndarray:
    """Render pupil (green) and iris (red) circles on an RGB image for visual inspection."""
    if gray.ndim == 2:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        vis = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

    p_center = (int(round(seg.pupil_x)), int(round(seg.pupil_y)))
    i_center = (int(round(seg.iris_x)), int(round(seg.iris_y)))

    cv2.circle(vis, p_center, max(1, int(round(seg.pupil_r))), (0, 220, 0), 2)
    cv2.circle(vis, i_center, max(1, int(round(seg.iris_r))), (230, 40, 40), 2)
    cv2.drawMarker(vis, p_center, (0, 140, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    cv2.drawMarker(vis, i_center, (255, 220, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
    return vis
