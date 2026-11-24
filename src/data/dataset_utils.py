"""Utilities for scanning, validating, and splitting CASIA-Iris-Thousand."""

from __future__ import annotations

import itertools
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


# Matches e.g. S5000L00.jpg -> subject="000", eye="L", sample="00"
_FILENAME_PATTERN = re.compile(r"^S\d(\d{3})([LR])(\d+)$", re.IGNORECASE)


def _parse_casia_path(rel_path: Path) -> tuple[str, str, int]:
    """Extract (subject_id, eye_side, sample_idx) from a CASIA-Iris-Thousand relative path."""
    stem = rel_path.stem
    match = _FILENAME_PATTERN.match(stem)
    if match:
        subject_id = match.group(1)
        eye_side = match.group(2).upper()
        sample_idx = int(match.group(3))
        return subject_id, eye_side, sample_idx

    parts = rel_path.parts
    if len(parts) >= 3:
        eye_candidate = parts[-2].upper()
        subj_candidate = parts[-3]
        if eye_candidate in {"L", "R"} and subj_candidate.isdigit():
            subj_num = int(subj_candidate)
            subject_id = f"{subj_num:03d}" if subj_num <= 999 else str(subj_num)
            trailing_digits = re.search(r"(\d+)$", stem)
            sample_idx = int(trailing_digits.group(1)) if trailing_digits else 0
            return subject_id, eye_candidate, sample_idx

    raise ValueError(
        f"Could not parse subject_id and eye_side from relative path: '{rel_path.as_posix()}'. "
        "Expected filename like 'S5000L00.jpg' or directory layout '<subject>/<L|R>/<file>'."
    )


def scan_casia_thousand(
    raw_dir: str | Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".bmp", ".png"),
) -> pd.DataFrame:
    """Recursively scan CASIA-Iris-Thousand and return a deterministic metadata DataFrame."""
    raw_path = Path(raw_dir)
    if not raw_path.exists() or not raw_path.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_path}")

    allowed_exts = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
    }

    records: list[dict[str, object]] = []
    for file_path in raw_path.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(raw_path)
        if any(part == "__MACOSX" or part.startswith(".") for part in rel_path.parts):
            continue
        if file_path.suffix.lower() not in allowed_exts:
            continue

        subject_id, eye_side, sample_idx = _parse_casia_path(rel_path)
        records.append(
            {
                "rel_path": rel_path.as_posix(),
                "filename": file_path.name,
                "subject_id": subject_id,
                "eye_side": eye_side,
                "eye_id": f"{subject_id}_{eye_side}",
                "sample_idx": sample_idx,
            }
        )

    if not records:
        raise ValueError(
            f"No image files with extensions {sorted(allowed_exts)} found under '{raw_path}'."
        )

    df = pd.DataFrame(records)
    df = df.sort_values(
        by=["subject_id", "eye_side", "filename"], kind="mergesort"
    ).reset_index(drop=True)

    sorted_eyes = sorted(df["eye_id"].unique())
    eye_to_idx = {eye_id: idx for idx, eye_id in enumerate(sorted_eyes)}
    df["class_idx"] = df["eye_id"].map(eye_to_idx)
    return df


def validate_images(
    df: pd.DataFrame,
    raw_dir: str | Path,
    check_pixel_stats: bool = True,
    specular_threshold: int = 245,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Verify image readability, dimensions, and optional grayscale quality statistics."""
    raw_path = Path(raw_dir)
    out = df.copy().reset_index(drop=True)

    widths: list[int] = []
    heights: list[int] = []
    channels_list: list[int] = []
    file_sizes: list[int] = []
    is_valid_list: list[bool] = []
    error_msgs: list[str] = []
    mean_intensities: list[float] = []
    std_intensities: list[float] = []
    specular_ratios: list[float] = []
    laplacian_vars: list[float] = []

    iterator = out["rel_path"].tolist()
    if show_progress:
        iterator = tqdm(iterator, desc="Validating images", total=len(out))

    for rel_path in iterator:
        img_path = raw_path / rel_path
        file_size = img_path.stat().st_size if img_path.exists() else 0
        file_sizes.append(int(file_size))

        try:
            with Image.open(img_path) as img:
                img.load()
                width, height = img.size
                channels = len(img.getbands())

                if check_pixel_stats:
                    gray = img.convert("L")
                    arr = np.asarray(gray, dtype=np.float32)
                    mean_val = float(arr.mean())
                    std_val = float(arr.std())
                    spec_val = float((arr >= specular_threshold).mean())
                    if arr.shape[0] >= 3 and arr.shape[1] >= 3:
                        lap = (
                            -4.0 * arr[1:-1, 1:-1]
                            + arr[:-2, 1:-1]
                            + arr[2:, 1:-1]
                            + arr[1:-1, :-2]
                            + arr[1:-1, 2:]
                        )
                        lap_var = float(lap.var())
                    else:
                        lap_var = 0.0
                else:
                    mean_val = float("nan")
                    std_val = float("nan")
                    spec_val = float("nan")
                    lap_var = float("nan")

            widths.append(int(width))
            heights.append(int(height))
            channels_list.append(int(channels))
            is_valid_list.append(True)
            error_msgs.append("")
            mean_intensities.append(mean_val)
            std_intensities.append(std_val)
            specular_ratios.append(spec_val)
            laplacian_vars.append(lap_var)
        except Exception as exc:  # noqa: BLE001 - capture any PIL/OS corruption error per file
            widths.append(0)
            heights.append(0)
            channels_list.append(0)
            is_valid_list.append(False)
            error_msgs.append(str(exc))
            mean_intensities.append(float("nan"))
            std_intensities.append(float("nan"))
            specular_ratios.append(float("nan"))
            laplacian_vars.append(float("nan"))

    out["width"] = widths
    out["height"] = heights
    out["channels"] = channels_list
    out["file_size_bytes"] = file_sizes
    out["is_valid"] = is_valid_list
    out["error_msg"] = error_msgs

    if check_pixel_stats:
        out["mean_intensity"] = mean_intensities
        out["std_intensity"] = std_intensities
        out["specular_ratio"] = specular_ratios
        out["laplacian_var"] = laplacian_vars

    return out


def _filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy filtered to valid rows if an 'is_valid' column is present."""
    if "is_valid" in df.columns:
        return df[df["is_valid"].astype(bool)].copy()
    return df.copy()


def _proportional_three_way_counts(
    total: int,
    target_train: int,
    target_val: int,
    target_test: int,
) -> tuple[int, int, int]:
    """Allocate `total >= 3` items into train/val/test while keeping every split non-empty."""
    target_sum = target_train + target_val + target_test
    if total == target_sum:
        return target_train, target_val, target_test

    n_val = max(1, int(round(total * target_val / target_sum)))
    n_test = max(1, int(round(total * target_test / target_sum)))
    while n_val + n_test > total - 1:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    n_train = total - n_val - n_test
    return n_train, n_val, n_test


def create_closed_set_splits(
    df: pd.DataFrame,
    train_per_eye: int = 6,
    val_per_eye: int = 2,
    test_per_eye: int = 2,
    subset_num_subjects: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic per-eye train/val/test splits for closed-set (1:N) recognition."""
    valid_df = _filter_valid(df)
    if valid_df.empty:
        raise ValueError("Cannot create closed-set splits from an empty DataFrame.")

    valid_df = valid_df.sort_values(
        by=["eye_id", "rel_path"], kind="mergesort"
    ).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    split_parts: list[pd.DataFrame] = []

    for eye_id in sorted(valid_df["eye_id"].unique()):
        eye_rows = valid_df[valid_df["eye_id"] == eye_id].copy()
        n = len(eye_rows)
        if n < 3:
            warnings.warn(
                f"Skipping eye_id='{eye_id}' with only {n} valid image(s); "
                "at least 3 are required for train/val/test.",
                stacklevel=2,
            )
            continue

        n_train, n_val, _ = _proportional_three_way_counts(
            n, train_per_eye, val_per_eye, test_per_eye
        )
        perm = rng.permutation(n)
        eye_rows = eye_rows.iloc[perm].copy()

        split_labels = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
        eye_rows["split"] = split_labels
        split_parts.append(eye_rows)

    if not split_parts:
        raise ValueError("No eyes had enough images (>= 3) to create closed-set splits.")

    out = pd.concat(split_parts, ignore_index=True)
    sorted_subjects = sorted(out["subject_id"].astype(str).unique())
    subset_subjects = set(sorted_subjects[:subset_num_subjects])
    out["in_subset_200_eyes"] = out["subject_id"].astype(str).isin(subset_subjects)

    out = out.sort_values(
        by=["split", "subject_id", "eye_side", "filename"], kind="mergesort"
    ).reset_index(drop=True)
    return out


def create_open_set_splits(
    df: pd.DataFrame,
    train_subjects: int = 700,
    val_subjects: int = 150,
    test_subjects: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """Create subject-disjoint train/val/test splits for open-set (1:1) verification."""
    valid_df = _filter_valid(df)
    if valid_df.empty:
        raise ValueError("Cannot create open-set splits from an empty DataFrame.")

    valid_df = valid_df.copy()
    valid_df["subject_id"] = valid_df["subject_id"].astype(str)
    subjects = sorted(valid_df["subject_id"].unique())
    n_subj = len(subjects)
    if n_subj < 3:
        raise ValueError(
            f"At least 3 subjects are required for open-set splits, found {n_subj}."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_subj)
    shuffled_subjects = [subjects[i] for i in perm]

    n_train, n_val, _ = _proportional_three_way_counts(
        n_subj, train_subjects, val_subjects, test_subjects
    )

    subject_to_split: dict[str, str] = {}
    for subj in shuffled_subjects[:n_train]:
        subject_to_split[subj] = "train"
    for subj in shuffled_subjects[n_train : n_train + n_val]:
        subject_to_split[subj] = "val"
    for subj in shuffled_subjects[n_train + n_val :]:
        subject_to_split[subj] = "test"

    valid_df["split"] = valid_df["subject_id"].map(subject_to_split)

    # Sanity check: ensure zero subject or eye overlap across splits
    split_subjects = {
        s: set(valid_df.loc[valid_df["split"] == s, "subject_id"])
        for s in ("train", "val", "test")
    }
    split_eyes = {
        s: set(valid_df.loc[valid_df["split"] == s, "eye_id"])
        for s in ("train", "val", "test")
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert split_subjects[a].isdisjoint(split_subjects[b]), f"Subject overlap: {a} vs {b}"
        assert split_eyes[a].isdisjoint(split_eyes[b]), f"Eye overlap: {a} vs {b}"

    return valid_df.sort_values(
        by=["split", "subject_id", "eye_side", "filename"], kind="mergesort"
    ).reset_index(drop=True)


def generate_verification_pairs(
    split_df: pd.DataFrame,
    num_genuine: int = 3000,
    num_impostor: int = 3000,
    same_subject_impostor_ratio: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic genuine and impostor pairs from a single open-set split."""
    valid_df = _filter_valid(split_df)
    if valid_df.empty:
        raise ValueError("Cannot generate verification pairs from an empty DataFrame.")

    valid_df = valid_df.sort_values(by=["rel_path"], kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng(seed)

    paths = valid_df["rel_path"].to_numpy()
    eye_ids = valid_df["eye_id"].to_numpy()
    subj_ids = valid_df["subject_id"].astype(str).to_numpy()

    # 1. Enumerate all genuine pairs (same eye_id, distinct images)
    all_genuine: list[tuple[int, int]] = []
    for eye_id in sorted(valid_df["eye_id"].unique()):
        idxs = np.flatnonzero(eye_ids == eye_id)
        if len(idxs) >= 2:
            all_genuine.extend(itertools.combinations(idxs.tolist(), 2))

    if not all_genuine:
        raise ValueError("No valid genuine pairs available in split_df.")

    actual_genuine = min(num_genuine, len(all_genuine))
    if actual_genuine < num_genuine:
        warnings.warn(
            f"Requested {num_genuine} genuine pairs, but only {len(all_genuine)} exist; "
            f"capping to {actual_genuine}.",
            stacklevel=2,
        )
    chosen_gen_idx = rng.choice(len(all_genuine), size=actual_genuine, replace=False)
    genuine_pairs = [all_genuine[int(i)] for i in chosen_gen_idx]

    # 2. Same-subject opposite-eye impostor pairs (hard negative pairs)
    target_same_subj = int(round(num_impostor * same_subject_impostor_ratio))
    all_same_subj_impostor: list[tuple[int, int]] = []
    for subj_id in sorted(np.unique(subj_ids)):
        subj_mask = subj_ids == subj_id
        subj_eyes = sorted(np.unique(eye_ids[subj_mask]))
        if len(subj_eyes) < 2:
            continue
        for eye_a, eye_b in itertools.combinations(subj_eyes, 2):
            idxs_a = np.flatnonzero(eye_ids == eye_a)
            idxs_b = np.flatnonzero(eye_ids == eye_b)
            for ia, ib in itertools.product(idxs_a.tolist(), idxs_b.tolist()):
                all_same_subj_impostor.append((min(ia, ib), max(ia, ib)))

    actual_same_subj = min(target_same_subj, len(all_same_subj_impostor))
    if actual_same_subj > 0:
        chosen_same_idx = rng.choice(
            len(all_same_subj_impostor), size=actual_same_subj, replace=False
        )
        same_subj_pairs = [all_same_subj_impostor[int(i)] for i in chosen_same_idx]
    else:
        same_subj_pairs = []

    # 3. Different-subject impostor pairs
    unique_subjects = sorted(np.unique(subj_ids))
    if len(unique_subjects) < 2:
        raise ValueError("At least 2 distinct subjects are required to generate impostor pairs.")

    subj_to_indices = [np.flatnonzero(subj_ids == s) for s in unique_subjects]
    subj_counts = np.array([len(g) for g in subj_to_indices], dtype=np.int64)
    total_possible_diff = int(
        (subj_counts.sum() ** 2 - np.square(subj_counts).sum()) // 2
    )
    target_diff_subj = min(num_impostor - actual_same_subj, total_possible_diff)

    seen_diff: set[tuple[int, int]] = set()
    diff_subj_pairs: list[tuple[int, int]] = []

    # Weight subjects by image count for uniform sampling across image pairs
    subj_weights = subj_counts / subj_counts.sum()
    max_attempts = max(10000, target_diff_subj * 50)
    attempts = 0

    while len(diff_subj_pairs) < target_diff_subj and attempts < max_attempts:
        attempts += 1
        s1, s2 = rng.choice(len(unique_subjects), size=2, replace=False, p=subj_weights)
        i1 = int(rng.choice(subj_to_indices[int(s1)]))
        i2 = int(rng.choice(subj_to_indices[int(s2)]))
        pair = (min(i1, i2), max(i1, i2))
        if pair not in seen_diff:
            seen_diff.add(pair)
            diff_subj_pairs.append(pair)

    # Deterministic fallback when sampling close to exhaustion on tiny datasets
    if len(diff_subj_pairs) < target_diff_subj:
        remaining: list[tuple[int, int]] = []
        for s1_idx, s2_idx in itertools.combinations(range(len(unique_subjects)), 2):
            for i1, i2 in itertools.product(
                subj_to_indices[s1_idx].tolist(), subj_to_indices[s2_idx].tolist()
            ):
                pair = (min(i1, i2), max(i1, i2))
                if pair not in seen_diff:
                    remaining.append(pair)
        needed = min(target_diff_subj - len(diff_subj_pairs), len(remaining))
        if needed > 0:
            extra_idx = rng.choice(len(remaining), size=needed, replace=False)
            for idx in extra_idx:
                diff_subj_pairs.append(remaining[int(idx)])

    records: list[dict[str, object]] = []
    for i1, i2 in genuine_pairs:
        records.append(
            {
                "img1_path": paths[i1],
                "img2_path": paths[i2],
                "eye1_id": eye_ids[i1],
                "eye2_id": eye_ids[i2],
                "subject1_id": subj_ids[i1],
                "subject2_id": subj_ids[i2],
                "label": 1,
                "pair_type": "genuine",
            }
        )
    for i1, i2 in same_subj_pairs:
        records.append(
            {
                "img1_path": paths[i1],
                "img2_path": paths[i2],
                "eye1_id": eye_ids[i1],
                "eye2_id": eye_ids[i2],
                "subject1_id": subj_ids[i1],
                "subject2_id": subj_ids[i2],
                "label": 0,
                "pair_type": "impostor_same_subject",
            }
        )
    for i1, i2 in diff_subj_pairs:
        records.append(
            {
                "img1_path": paths[i1],
                "img2_path": paths[i2],
                "eye1_id": eye_ids[i1],
                "eye2_id": eye_ids[i2],
                "subject1_id": subj_ids[i1],
                "subject2_id": subj_ids[i2],
                "label": 0,
                "pair_type": "impostor_diff_subject",
            }
        )

    pairs_df = pd.DataFrame(records)
    perm = rng.permutation(len(pairs_df))
    pairs_df = pairs_df.iloc[perm].reset_index(drop=True)
    pairs_df.insert(0, "pair_id", np.arange(len(pairs_df), dtype=int))
    return pairs_df


def _validate_pair_frame(
    pairs_df: pd.DataFrame,
    allowed_paths: set[str],
    name: str,
) -> set[tuple[str, str]]:
    """Assert structural integrity and return canonical unordered path pairs."""
    p1 = set(pairs_df["img1_path"])
    p2 = set(pairs_df["img2_path"])
    assert p1.issubset(allowed_paths), f"{name}: img1_path contains images outside split."
    assert p2.issubset(allowed_paths), f"{name}: img2_path contains images outside split."
    assert (pairs_df["img1_path"] != pairs_df["img2_path"]).all(), (
        f"{name}: found self-comparison pair(s)."
    )

    gen = pairs_df[pairs_df["label"] == 1]
    imp = pairs_df[pairs_df["label"] == 0]
    assert (gen["eye1_id"] == gen["eye2_id"]).all(), (
        f"{name}: genuine pair has mismatched eye IDs."
    )
    assert (imp["eye1_id"] != imp["eye2_id"]).all(), (
        f"{name}: impostor pair has identical eye IDs."
    )

    canonical = {
        (a, b) if a <= b else (b, a)
        for a, b in zip(pairs_df["img1_path"], pairs_df["img2_path"])
    }
    assert len(canonical) == len(pairs_df), f"{name}: contains duplicate image pairs."
    return canonical


def verify_no_leakage(
    closed_df: pd.DataFrame,
    open_df: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
) -> dict[str, object]:
    """Verify zero data leakage across closed-set, open-set, and verification pair files."""
    # 1. Closed-set checks
    closed_paths = {
        s: set(closed_df.loc[closed_df["split"] == s, "rel_path"])
        for s in ("train", "val", "test")
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert closed_paths[a].isdisjoint(closed_paths[b]), (
            f"Closed-set image leakage between {a} and {b}."
        )

    closed_eyes = {
        s: set(closed_df.loc[closed_df["split"] == s, "eye_id"])
        for s in ("train", "val", "test")
    }
    assert closed_eyes["val"].issubset(closed_eyes["train"]), (
        "Closed-set val contains eyes missing from train."
    )
    assert closed_eyes["test"].issubset(closed_eyes["train"]), (
        "Closed-set test contains eyes missing from train."
    )

    # 2. Open-set checks
    open_paths = {
        s: set(open_df.loc[open_df["split"] == s, "rel_path"])
        for s in ("train", "val", "test")
    }
    open_subjects = {
        s: set(open_df.loc[open_df["split"] == s, "subject_id"].astype(str))
        for s in ("train", "val", "test")
    }
    open_eyes = {
        s: set(open_df.loc[open_df["split"] == s, "eye_id"])
        for s in ("train", "val", "test")
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert open_paths[a].isdisjoint(open_paths[b]), (
            f"Open-set image leakage between {a} and {b}."
        )
        assert open_subjects[a].isdisjoint(open_subjects[b]), (
            f"Open-set subject leakage between {a} and {b}."
        )
        assert open_eyes[a].isdisjoint(open_eyes[b]), (
            f"Open-set eye leakage between {a} and {b}."
        )

    # 3. Verification pair checks
    val_pair_set = _validate_pair_frame(val_pairs, open_paths["val"], "val_pairs")
    test_pair_set = _validate_pair_frame(test_pairs, open_paths["test"], "test_pairs")
    assert val_pair_set.isdisjoint(test_pair_set), (
        "Pair leakage detected between val_pairs and test_pairs."
    )

    return {
        "passed": True,
        "closed_set_counts": {k: len(v) for k, v in closed_paths.items()},
        "closed_set_num_eyes": len(closed_eyes["train"]),
        "open_set_image_counts": {k: len(v) for k, v in open_paths.items()},
        "open_set_subject_counts": {k: len(v) for k, v in open_subjects.items()},
        "val_pairs_count": len(val_pairs),
        "test_pairs_count": len(test_pairs),
    }


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


def build_dataset_summary(
    meta_df: pd.DataFrame,
    closed_df: pd.DataFrame,
    open_df: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    leakage_report: dict[str, object],
    config: dict[str, object],
) -> dict[str, object]:
    """Build a JSON-serializable summary of dataset quality and splits."""
    valid_df = _filter_valid(meta_df)
    res_counts = (
        valid_df.assign(
            resolution=valid_df["width"].astype(str) + "x" + valid_df["height"].astype(str)
        )["resolution"]
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
