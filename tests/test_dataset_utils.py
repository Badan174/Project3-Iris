"""Unit tests for CASIA-Iris-Thousand scanning, validation, splitting, and CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    build_dataset_summary,
    create_closed_set_splits,
    create_open_set_splits,
    generate_verification_pairs,
    scan_casia_thousand,
    validate_images,
    verify_no_leakage,
)


def _create_synthetic_casia_tree(
    root: Path,
    num_subjects: int = 20,
    samples_per_eye: int = 10,
) -> dict[str, Path]:
    """Create a synthetic CASIA-Iris-Thousand directory tree for unit testing."""
    special_paths: dict[str, Path] = {}
    rng = np.random.default_rng(123)

    for subj_idx in range(num_subjects):
        subj_str = f"{subj_idx:03d}"
        for side in ("L", "R"):
            eye_dir = root / subj_str / side
            eye_dir.mkdir(parents=True, exist_ok=True)
            for sample_idx in range(samples_per_eye):
                filename = f"S5{subj_str}{side}{sample_idx:02d}.jpg"
                out_path = eye_dir / filename

                if subj_idx == 0 and side == "L" and sample_idx == 0:
                    # Bright glare image (high specular_ratio)
                    arr = np.full((48, 64), 255, dtype=np.uint8)
                    special_paths["glare"] = out_path
                elif subj_idx == 0 and side == "L" and sample_idx == 1:
                    # Completely flat mid-gray image (laplacian_var == 0)
                    arr = np.full((48, 64), 100, dtype=np.uint8)
                    special_paths["flat"] = out_path
                elif subj_idx == 0 and side == "L" and sample_idx == 2:
                    # Sharp checkerboard image (high laplacian_var, low glare)
                    yy, xx = np.indices((48, 64))
                    arr = np.where((yy + xx) % 2 == 0, 20, 200).astype(np.uint8)
                    special_paths["sharp"] = out_path
                else:
                    arr = rng.integers(30, 190, size=(48, 64), dtype=np.uint8)

                Image.fromarray(arr, mode="L").save(out_path, quality=95)

    # Add macOS junk & hidden files that must be ignored during scanning
    macosx_dir = root / "__MACOSX" / "000" / "L"
    macosx_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(
        macosx_dir / "._S5000L00.jpg"
    )
    hidden_file = root / "000" / "L" / "._S5000L00.jpg"
    hidden_file.write_bytes(b"junk")

    return special_paths


class TestDatasetUtils(unittest.TestCase):
    """Test suite for src.data.dataset_utils and scripts/prepare_splits.py."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.raw_dir = self.root / "CASIA-Iris-Thousand"
        self.special = _create_synthetic_casia_tree(
            self.raw_dir, num_subjects=20, samples_per_eye=10
        )

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_scan_casia_thousand(self) -> None:
        df = scan_casia_thousand(self.raw_dir)
        self.assertEqual(len(df), 400)
        self.assertEqual(df["subject_id"].nunique(), 20)
        self.assertEqual(df["eye_id"].nunique(), 40)
        self.assertEqual(df["class_idx"].min(), 0)
        self.assertEqual(df["class_idx"].max(), 39)
        self.assertTrue(all("\\" not in p for p in df["rel_path"]))
        self.assertFalse(any("__MACOSX" in p for p in df["rel_path"]))
        self.assertIn("000_L", set(df["eye_id"]))
        self.assertIn("019_R", set(df["eye_id"]))

    def test_validate_images_stats_and_corruption(self) -> None:
        df = scan_casia_thousand(self.raw_dir)
        validated = validate_images(
            df, self.raw_dir, check_pixel_stats=True, specular_threshold=245
        )
        self.assertTrue(validated["is_valid"].all())
        self.assertTrue((validated["width"] == 64).all())
        self.assertTrue((validated["height"] == 48).all())

        glare_rel = self.special["glare"].relative_to(self.raw_dir).as_posix()
        flat_rel = self.special["flat"].relative_to(self.raw_dir).as_posix()
        sharp_rel = self.special["sharp"].relative_to(self.raw_dir).as_posix()

        glare_row = validated.loc[validated["rel_path"] == glare_rel].iloc[0]
        flat_row = validated.loc[validated["rel_path"] == flat_rel].iloc[0]
        sharp_row = validated.loc[validated["rel_path"] == sharp_rel].iloc[0]

        self.assertGreater(glare_row["specular_ratio"], flat_row["specular_ratio"])
        self.assertGreater(sharp_row["laplacian_var"], flat_row["laplacian_var"])

        # Corrupt one image and confirm validation flags it gracefully
        target_corrupt = self.raw_dir / "001" / "L" / "S5001L00.jpg"
        target_corrupt.write_bytes(b"not_a_valid_jpeg")
        validated_with_err = validate_images(df, self.raw_dir, check_pixel_stats=True)
        bad_rows = validated_with_err[~validated_with_err["is_valid"]]
        self.assertEqual(len(bad_rows), 1)
        self.assertEqual(bad_rows.iloc[0]["filename"], "S5001L00.jpg")

    def test_create_closed_set_splits_deterministic(self) -> None:
        df = validate_images(scan_casia_thousand(self.raw_dir), self.raw_dir)
        split1 = create_closed_set_splits(
            df, train_per_eye=6, val_per_eye=2, test_per_eye=2, subset_num_subjects=5, seed=42
        )
        split2 = create_closed_set_splits(
            df, train_per_eye=6, val_per_eye=2, test_per_eye=2, subset_num_subjects=5, seed=42
        )
        pd.testing.assert_frame_equal(split1, split2)

        counts = split1["split"].value_counts().to_dict()
        self.assertEqual(counts, {"train": 240, "val": 80, "test": 80})

        per_eye_split = split1.groupby(["eye_id", "split"]).size().unstack(fill_value=0)
        self.assertTrue((per_eye_split["train"] == 6).all())
        self.assertTrue((per_eye_split["val"] == 2).all())
        self.assertTrue((per_eye_split["test"] == 2).all())

        # First 5 subjects * 2 eyes * 10 images = 100 images in subset
        self.assertEqual(int(split1["in_subset_200_eyes"].sum()), 100)

    def test_create_open_set_splits_disjoint_and_deterministic(self) -> None:
        df = validate_images(scan_casia_thousand(self.raw_dir), self.raw_dir)
        open1 = create_open_set_splits(
            df, train_subjects=700, val_subjects=150, test_subjects=150, seed=42
        )
        open2 = create_open_set_splits(
            df, train_subjects=700, val_subjects=150, test_subjects=150, seed=42
        )
        pd.testing.assert_frame_equal(open1, open2)

        subj_sets = {
            s: set(open1.loc[open1["split"] == s, "subject_id"])
            for s in ("train", "val", "test")
        }
        self.assertEqual(len(subj_sets["train"]) + len(subj_sets["val"]) + len(subj_sets["test"]), 20)
        self.assertTrue(subj_sets["train"].isdisjoint(subj_sets["val"]))
        self.assertTrue(subj_sets["train"].isdisjoint(subj_sets["test"]))
        self.assertTrue(subj_sets["val"].isdisjoint(subj_sets["test"]))

    def test_generate_verification_pairs_and_leakage_checks(self) -> None:
        df = validate_images(scan_casia_thousand(self.raw_dir), self.raw_dir)
        closed_df = create_closed_set_splits(df, subset_num_subjects=5, seed=42)
        open_df = create_open_set_splits(df, seed=42)

        val_pairs_1 = generate_verification_pairs(
            open_df[open_df["split"] == "val"],
            num_genuine=50,
            num_impostor=50,
            same_subject_impostor_ratio=0.25,
            seed=43,
        )
        val_pairs_2 = generate_verification_pairs(
            open_df[open_df["split"] == "val"],
            num_genuine=50,
            num_impostor=50,
            same_subject_impostor_ratio=0.25,
            seed=43,
        )
        pd.testing.assert_frame_equal(val_pairs_1, val_pairs_2)

        test_pairs = generate_verification_pairs(
            open_df[open_df["split"] == "test"],
            num_genuine=50,
            num_impostor=50,
            same_subject_impostor_ratio=0.25,
            seed=44,
        )

        self.assertEqual(len(val_pairs_1), 100)
        self.assertEqual(int((val_pairs_1["label"] == 1).sum()), 50)
        self.assertEqual(int((val_pairs_1["label"] == 0).sum()), 50)
        self.assertIn("impostor_same_subject", set(val_pairs_1["pair_type"]))
        self.assertIn("impostor_diff_subject", set(val_pairs_1["pair_type"]))

        report = verify_no_leakage(closed_df, open_df, val_pairs_1, test_pairs)
        self.assertTrue(report["passed"])

        summary = build_dataset_summary(
            df,
            closed_df,
            open_df,
            val_pairs_1,
            test_pairs,
            report,
            {"dataset_name": "Test", "seed": 42},
        )
        for key in (
            "resolution_distribution",
            "quality_stats",
            "closed_set",
            "open_set",
            "verification_pairs",
            "leakage_check",
        ):
            self.assertIn(key, summary)

        # Deliberately inject leakage from val into test_pairs and verify check fails
        leaked_test = test_pairs.copy()
        leaked_test.iloc[0, leaked_test.columns.get_loc("img1_path")] = val_pairs_1.iloc[0][
            "img1_path"
        ]
        with self.assertRaises(AssertionError):
            verify_no_leakage(closed_df, open_df, val_pairs_1, leaked_test)

    def test_prepare_splits_cli_end_to_end(self) -> None:
        cfg_path = self.root / "test_config.yaml"
        out_dir = self.root / "splits_out"
        test_cfg = {
            "dataset_name": "CASIA-Iris-Thousand-Synthetic",
            "seed": 42,
            "image_extensions": [".jpg"],
            "closed_set": {
                "train_per_eye": 6,
                "val_per_eye": 2,
                "test_per_eye": 2,
                "subset_num_subjects": 5,
            },
            "open_set": {
                "train_subjects": 14,
                "val_subjects": 3,
                "test_subjects": 3,
            },
            "verification_pairs": {
                "val_num_genuine": 40,
                "val_num_impostor": 40,
                "test_num_genuine": 40,
                "test_num_impostor": 40,
                "same_subject_impostor_ratio": 0.25,
            },
            "quality_thresholds": {"specular_pixel_threshold": 245},
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(test_cfg, f)

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "prepare_splits.py"),
            "--raw-dir",
            str(self.raw_dir),
            "--config",
            str(cfg_path),
            "--output-dir",
            str(out_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=f"CLI failed:\n{proc.stdout}\n{proc.stderr}")

        expected_files = [
            "metadata.csv",
            "closed_set_splits.csv",
            "open_set_splits.csv",
            "val_pairs.csv",
            "test_pairs.csv",
            "dataset_summary.json",
        ]
        for fname in expected_files:
            self.assertTrue((out_dir / fname).is_file(), f"Missing output file: {fname}")

        with open(out_dir / "dataset_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["valid_images"], 400)
        self.assertTrue(summary["leakage_check"]["passed"])
        for key in (
            "resolution_distribution",
            "quality_stats",
            "closed_set",
            "open_set",
            "verification_pairs",
        ):
            self.assertIn(key, summary)


if __name__ == "__main__":
    unittest.main()
