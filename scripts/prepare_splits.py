"""CLI script to scan CASIA-Iris-Thousand, validate images, and generate fixed splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    create_closed_set_splits,
    create_open_set_splits,
    generate_verification_pairs,
    scan_casia_thousand,
    validate_images,
    verify_no_leakage,
)


def _series_summary(series: pd.Series) -> dict[str, float]:
    """Compute descriptive summary stats for a numeric Series."""
    clean = series.dropna()
    if clean.empty:
        return {}
    return {
        "mean": round(float(clean.mean()), 6),
        "std": round(float(clean.std()), 6),
        "min": round(float(clean.min()), 6),
        "p25": round(float(clean.quantile(0.25)), 6),
        "median": round(float(clean.median()), 6),
        "p75": round(float(clean.quantile(0.75)), 6),
        "max": round(float(clean.max()), 6),
    }


def _build_summary(
    meta_df: pd.DataFrame,
    closed_df: pd.DataFrame,
    open_df: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    leakage_report: dict[str, object],
    config: dict[str, object],
) -> dict[str, object]:
    """Build a JSON-serializable summary of dataset quality and splits."""
    valid_df = meta_df[meta_df["is_valid"].astype(bool)]
    res_counts = (
        valid_df.assign(resolution=valid_df["width"].astype(str) + "x" + valid_df["height"].astype(str))["resolution"]
        .value_counts()
        .to_dict()
    )
    eye_counts = valid_df.groupby("eye_id").size()

    quality_stats: dict[str, dict[str, float]] = {}
    for col in ("mean_intensity", "std_intensity", "specular_ratio", "laplacian_var"):
        if col in valid_df.columns:
            quality_stats[col] = _series_summary(valid_df[col])

    return {
        "dataset_name": config.get("dataset_name", "CASIA-Iris-Thousand"),
        "seed": int(config.get("seed", 42)),
        "total_images_scanned": int(len(meta_df)),
        "valid_images": int(len(valid_df)),
        "invalid_images": int(len(meta_df) - len(valid_df)),
        "num_subjects": int(valid_df["subject_id"].nunique()),
        "num_eyes": int(valid_df["eye_id"].nunique()),
        "images_per_eye_min": int(eye_counts.min()) if not eye_counts.empty else 0,
        "images_per_eye_max": int(eye_counts.max()) if not eye_counts.empty else 0,
        "resolution_distribution": {str(k): int(v) for k, v in res_counts.items()},
        "quality_stats": quality_stats,
        "closed_set": {
            "split_counts": {
                str(k): int(v) for k, v in closed_df["split"].value_counts().to_dict().items()
            },
            "subset_200_eyes_images": int(closed_df["in_subset_200_eyes"].sum()),
            "subset_200_eyes_classes": int(
                closed_df.loc[closed_df["in_subset_200_eyes"], "eye_id"].nunique()
            ),
        },
        "open_set": {
            "image_counts": {
                str(k): int(v) for k, v in open_df["split"].value_counts().to_dict().items()
            },
            "subject_counts": {
                str(k): int(v)
                for k, v in open_df.groupby("split")["subject_id"].nunique().to_dict().items()
            },
        },
        "verification_pairs": {
            "val_pair_types": {
                str(k): int(v) for k, v in val_pairs["pair_type"].value_counts().to_dict().items()
            },
            "test_pair_types": {
                str(k): int(v) for k, v in test_pairs["pair_type"].value_counts().to_dict().items()
            },
        },
        "leakage_check": leakage_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan CASIA-Iris-Thousand, validate images, and create fixed splits."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Path to extracted raw CASIA-Iris-Thousand root directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "dataset_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "splits",
        help="Directory where metadata, split CSVs, and summary JSON will be saved.",
    )
    parser.add_argument(
        "--skip-pixel-stats",
        action="store_true",
        help="Skip computing intensity/specular/Laplacian stats for faster validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("seed", 42))
    extensions = tuple(config.get("image_extensions", [".jpg", ".jpeg", ".bmp", ".png"]))
    closed_cfg = config.get("closed_set", {})
    open_cfg = config.get("open_set", {})
    pairs_cfg = config.get("verification_pairs", {})
    qual_cfg = config.get("quality_thresholds", {})

    print(f"[1/6] Scanning images in: {args.raw_dir}")
    scanned_df = scan_casia_thousand(args.raw_dir, extensions=extensions)
    print(
        f"      Found {len(scanned_df)} files | "
        f"{scanned_df['subject_id'].nunique()} subjects | "
        f"{scanned_df['eye_id'].nunique()} eyes"
    )

    print("[2/6] Validating image readability and quality stats...")
    meta_df = validate_images(
        scanned_df,
        args.raw_dir,
        check_pixel_stats=not args.skip_pixel_stats,
        specular_threshold=int(qual_cfg.get("specular_pixel_threshold", 245)),
        show_progress=True,
    )

    print("[3/6] Generating closed-set (1:N) splits...")
    closed_df = create_closed_set_splits(
        meta_df,
        train_per_eye=int(closed_cfg.get("train_per_eye", 6)),
        val_per_eye=int(closed_cfg.get("val_per_eye", 2)),
        test_per_eye=int(closed_cfg.get("test_per_eye", 2)),
        subset_num_subjects=int(closed_cfg.get("subset_num_subjects", 100)),
        seed=seed,
    )

    print("[4/6] Generating open-set (1:1) subject-disjoint splits...")
    open_df = create_open_set_splits(
        meta_df,
        train_subjects=int(open_cfg.get("train_subjects", 700)),
        val_subjects=int(open_cfg.get("val_subjects", 150)),
        test_subjects=int(open_cfg.get("test_subjects", 150)),
        seed=seed,
    )

    print("[5/6] Generating verification pairs (val & test)...")
    val_pairs = generate_verification_pairs(
        open_df[open_df["split"] == "val"],
        num_genuine=int(pairs_cfg.get("val_num_genuine", 3000)),
        num_impostor=int(pairs_cfg.get("val_num_impostor", 3000)),
        same_subject_impostor_ratio=float(pairs_cfg.get("same_subject_impostor_ratio", 0.25)),
        seed=seed + 1,
    )
    test_pairs = generate_verification_pairs(
        open_df[open_df["split"] == "test"],
        num_genuine=int(pairs_cfg.get("test_num_genuine", 3000)),
        num_impostor=int(pairs_cfg.get("test_num_impostor", 3000)),
        same_subject_impostor_ratio=float(pairs_cfg.get("same_subject_impostor_ratio", 0.25)),
        seed=seed + 2,
    )

    print("[6/6] Verifying zero data leakage and writing outputs...")
    leakage_report = verify_no_leakage(closed_df, open_df, val_pairs, test_pairs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(args.output_dir / "metadata.csv", index=False)
    closed_df.to_csv(args.output_dir / "closed_set_splits.csv", index=False)
    open_df.to_csv(args.output_dir / "open_set_splits.csv", index=False)
    val_pairs.to_csv(args.output_dir / "val_pairs.csv", index=False)
    test_pairs.to_csv(args.output_dir / "test_pairs.csv", index=False)

    summary = _build_summary(
        meta_df, closed_df, open_df, val_pairs, test_pairs, leakage_report, config
    )
    with open(args.output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== DATASET & SPLIT SUMMARY ===")
    print(f"Total valid images   : {summary['valid_images']} / {summary['total_images_scanned']}")
    print(f"Subjects / Eye classes: {summary['num_subjects']} / {summary['num_eyes']}")
    print(f"Resolutions          : {summary['resolution_distribution']}")
    print(f"Closed-set (1:N)     : {summary['closed_set']['split_counts']}")
    print(f"Open-set subjects    : {summary['open_set']['subject_counts']}")
    print(f"Val pairs            : {summary['verification_pairs']['val_pair_types']}")
    print(f"Test pairs           : {summary['verification_pairs']['test_pair_types']}")
    print(f"Leakage check passed : {leakage_report['passed']}")
    print(f"Saved files to       : {args.output_dir}")


if __name__ == "__main__":
    main()
