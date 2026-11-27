"""Unit tests for Stage 2 iris preprocessing (segmentation, Rubber Sheet, ROI, pipeline)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import (  # noqa: E402
    SegmentationResult,
    archive_processed_dir,
    detect_and_inpaint_specular,
    draw_segmentation_overlay,
    extract_iris_roi,
    normalize_rubber_sheet,
    preprocess_dataset,
    segment_iris,
)


def _make_synthetic_eye_image(
    width: int = 640,
    height: int = 480,
    px: float = 324.0,
    py: float = 242.0,
    pr: float = 36.0,
    ix: float = 318.0,
    iy: float = 238.0,
    ir: float = 108.0,
    add_glint: bool = True,
    add_frame_bar: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Generate a synthetic 640x480 NIR eyeball image with known ground-truth circles."""
    yy, xx = np.indices((height, width), dtype=np.float32)

    # Sclera background
    img = np.full((height, width), 192.0, dtype=np.float32)

    # Iris disk with radial + angular texture
    dist_iris = np.hypot(xx - ix, yy - iy)
    angle = np.arctan2(yy - iy, xx - ix)
    iris_tex = 105.0 + 12.0 * np.sin(8.0 * angle) + 8.0 * np.cos(0.35 * dist_iris)
    img[dist_iris <= ir] = iris_tex[dist_iris <= ir]

    # Dark pupil disk
    dist_pupil = np.hypot(xx - px, yy - py)
    img[dist_pupil <= pr] = 24.0

    # Optional dark vertical bar near left border simulating eyeglass frame
    if add_frame_bar:
        img[:, 8:20] = 15.0

    # Optional specular reflection hotspot inside iris texture
    glint_center = (int(round(px + pr + 20.0)), int(round(py)), 6)
    if add_glint:
        gx, gy, gr = glint_center
        glint_mask = np.hypot(xx - gx, yy - gy) <= gr
        img[glint_mask] = 255.0

    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0.7)
    return img, glint_center


class TestIrisPreprocessing(unittest.TestCase):
    """Test suite for iris segmentation, Rubber Sheet normalization, ROI crop, and batch pipeline."""

    def test_specular_detection_and_inpainting(self) -> None:
        img, (gx, gy, gr) = _make_synthetic_eye_image(add_glint=True)
        inpainted, spec_mask = detect_and_inpaint_specular(
            img, threshold=235, dilate_radius=3, inpaint_radius=4
        )
        self.assertEqual(inpainted.shape, img.shape)
        self.assertEqual(spec_mask.dtype, bool)
        self.assertTrue(spec_mask[gy, gx])
        # Glint hotspot should be smoothly filled below threshold
        self.assertLess(int(inpainted[gy, gx]), 200)

    def test_segment_iris_non_concentric_with_glint_and_frame(self) -> None:
        gt_px, gt_py, gt_pr = 324.0, 242.0, 36.0
        gt_ix, gt_iy, gt_ir = 318.0, 238.0, 108.0
        img, _ = _make_synthetic_eye_image(
            px=gt_px,
            py=gt_py,
            pr=gt_pr,
            ix=gt_ix,
            iy=gt_iy,
            ir=gt_ir,
            add_glint=True,
            add_frame_bar=True,
        )

        seg, _, _ = segment_iris(img)
        self.assertTrue(seg.success)
        self.assertNotEqual(seg.method, "fallback")
        self.assertLessEqual(np.hypot(seg.pupil_x - gt_px, seg.pupil_y - gt_py), 4.0)
        self.assertLessEqual(abs(seg.pupil_r - gt_pr), 4.0)
        self.assertLessEqual(np.hypot(seg.iris_x - gt_ix, seg.iris_y - gt_iy), 6.0)
        self.assertLessEqual(abs(seg.iris_r - gt_ir), 6.0)

        overlay = draw_segmentation_overlay(img, seg)
        self.assertEqual(overlay.shape, (480, 640, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_normalize_rubber_sheet_and_masking(self) -> None:
        img, _ = _make_synthetic_eye_image(
            px=324.0, py=242.0, pr=36.0, ix=318.0, iy=238.0, ir=108.0, add_glint=True
        )
        seg, _, spec_mask = segment_iris(img)
        polar_raw, polar_enh, valid_mask = normalize_rubber_sheet(
            img,
            seg,
            specular_mask=spec_mask,
            radial_res=64,
            angular_res=256,
            apply_clahe=True,
        )
        self.assertEqual(polar_raw.shape, (64, 256))
        self.assertEqual(polar_enh.shape, (64, 256))
        self.assertEqual(valid_mask.shape, (64, 256))
        self.assertEqual(polar_raw.dtype, np.uint8)
        self.assertEqual(polar_enh.dtype, np.uint8)
        self.assertEqual(valid_mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(valid_mask)).issubset({0, 255}), True)
        # Glint must cause some masked (0) pixels while most of the iris remains valid (255)
        self.assertIn(0, valid_mask)
        self.assertGreater((valid_mask > 0).mean(), 0.70)

    def test_extract_iris_roi_border_padding(self) -> None:
        img, _ = _make_synthetic_eye_image(
            px=65.0, py=60.0, pr=30.0, ix=60.0, iy=55.0, ir=100.0, add_glint=False, add_frame_bar=False
        )
        seg = SegmentationResult(
            pupil_x=65.0,
            pupil_y=60.0,
            pupil_r=30.0,
            iris_x=60.0,
            iris_y=55.0,
            iris_r=100.0,
            pupil_score=20.0,
            iris_score=15.0,
            method="contour_ido",
            success=True,
        )
        roi = extract_iris_roi(img, seg, margin_scale=1.15, output_size=224)
        self.assertEqual(roi.shape, (224, 224))
        self.assertEqual(roi.dtype, np.uint8)

    def test_preprocess_dataset_multiprocessing_and_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            out_dir = root / "processed"
            zip_path = root / "processed_test.zip"
            meta_csv = root / "metadata.csv"

            rows = []
            for subj_idx in range(2):
                subj_str = f"{subj_idx:03d}"
                for side in ("L", "R"):
                    eye_dir = raw_dir / subj_str / side
                    eye_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"S5{subj_str}{side}00.jpg"
                    img_path = eye_dir / fname
                    img, _ = _make_synthetic_eye_image(add_glint=True, add_frame_bar=True)
                    cv2.imwrite(str(img_path), img)
                    rows.append(
                        {
                            "rel_path": f"{subj_str}/{side}/{fname}",
                            "filename": fname,
                            "subject_id": subj_str,
                            "eye_side": side,
                            "eye_id": f"{subj_str}_{side}",
                            "sample_idx": 0,
                            "class_idx": subj_idx * 2 + (0 if side == "L" else 1),
                            "is_valid": True,
                        }
                    )

            pd.DataFrame(rows).to_csv(meta_csv, index=False)

            prev_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                returned_df = preprocess_dataset(
                    metadata_csv=meta_csv,
                    raw_dir=raw_dir,
                    output_dir=out_dir,
                    num_workers=2,
                    show_progress=False,
                )
            finally:
                os.chdir(prev_cwd)
            self.assertEqual(len(returned_df), 4)
            archive_processed_dir(out_dir, zip_path)

            seg_meta_path = out_dir / "segmentation_metadata.csv"
            summary_path = out_dir / "preprocessing_summary.json"
            self.assertTrue(seg_meta_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertTrue(zip_path.is_file())

            seg_df = pd.read_csv(seg_meta_path, dtype={"subject_id": str})
            self.assertEqual(len(seg_df), 4)
            self.assertTrue(seg_df["seg_success"].all())

            for _, row in seg_df.iterrows():
                rubber_img = cv2.imread(str(out_dir / row["rubber_sheet_path"]), cv2.IMREAD_GRAYSCALE)
                mask_img = cv2.imread(str(out_dir / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
                roi_img = cv2.imread(str(out_dir / row["roi_path"]), cv2.IMREAD_GRAYSCALE)
                self.assertEqual(rubber_img.shape, (64, 256))
                self.assertEqual(mask_img.shape, (64, 256))
                self.assertEqual(roi_img.shape, (224, 224))

            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["total_processed"], 4)
            self.assertEqual(summary["seg_success_count"], 4)

            with zipfile.ZipFile(zip_path, "r") as zf:
                names = set(zf.namelist())
            self.assertIn("segmentation_metadata.csv", names)
            self.assertIn("preprocessing_summary.json", names)
            self.assertIn("rubber_sheet/000/L/S5000L00.png", names)


if __name__ == "__main__":
    unittest.main()
