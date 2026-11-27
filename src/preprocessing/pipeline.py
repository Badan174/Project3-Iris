"""End-to-end preprocessing pipeline, batch multiprocessing, and archiving."""

from __future__ import annotations

import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import cv2
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable

from .iris_segmenter import merge_preprocessing_config, segment_iris
from .normalization import extract_iris_roi, normalize_rubber_sheet


def _init_worker() -> None:
    """Limit OpenCV per-process threads to avoid CPU oversubscription."""
    cv2.setNumThreads(1)


def preprocess_single_image(
    img_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run segmentation, Rubber Sheet normalization, and ROI extraction on one image."""
    merged_cfg = merge_preprocessing_config(cfg)
    norm_cfg = merged_cfg["normalization"]
    roi_cfg = merged_cfg["roi"]
    qual_cfg = merged_cfg["quality"]

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read grayscale image: {img_path}")

    seg, inpainted, specular_mask = segment_iris(gray, merged_cfg)
    polar_raw, polar_enhanced, valid_mask = normalize_rubber_sheet(
        gray,
        seg,
        specular_mask=specular_mask,
        radial_res=int(norm_cfg["radial_res"]),
        angular_res=int(norm_cfg["angular_res"]),
        apply_clahe=bool(norm_cfg["apply_clahe"]),
        clahe_clip_limit=float(norm_cfg["clahe_clip_limit"]),
        clahe_tile_grid=tuple(int(x) for x in norm_cfg["clahe_tile_grid"]),
    )
    roi = extract_iris_roi(
        gray,
        seg,
        margin_scale=float(roi_cfg["margin_scale"]),
        output_size=int(roi_cfg["output_size"]),
    )

    valid_pixel_ratio = float((valid_mask > 0).mean())
    seg_success = bool(seg.success and valid_pixel_ratio >= float(qual_cfg["min_valid_pixel_ratio"]))

    return {
        "gray": gray,
        "inpainted": inpainted,
        "specular_mask": specular_mask,
        "seg": seg,
        "polar_raw": polar_raw,
        "polar_enhanced": polar_enhanced,
        "valid_mask": valid_mask,
        "roi": roi,
        "valid_pixel_ratio": valid_pixel_ratio,
        "seg_success": seg_success,
    }


def _process_record_worker(task: tuple[dict[str, Any], str, str, dict[str, Any]]) -> dict[str, Any]:
    """Top-level picklable worker for ProcessPoolExecutor batch preprocessing."""
    row_dict, raw_dir_str, output_dir_str, cfg = task
    rel_path = str(row_dict["rel_path"])
    raw_dir = Path(raw_dir_str)
    output_dir = Path(output_dir_str)

    rel_no_ext = Path(rel_path).with_suffix("")
    rubber_rel = (Path("rubber_sheet") / rel_no_ext).with_suffix(".png")
    mask_rel = (Path("masks") / rel_no_ext).with_suffix(".png")
    roi_rel = (Path("roi") / rel_no_ext).with_suffix(".jpg")

    rubber_abs = output_dir / rubber_rel
    mask_abs = output_dir / mask_rel
    roi_abs = output_dir / roi_rel

    rubber_abs.parent.mkdir(parents=True, exist_ok=True)
    mask_abs.parent.mkdir(parents=True, exist_ok=True)
    roi_abs.parent.mkdir(parents=True, exist_ok=True)

    jpeg_quality = int(cfg["roi"]["jpeg_quality"])
    img_path = raw_dir / rel_path

    try:
        res = preprocess_single_image(img_path, cfg)
        seg = res["seg"]

        cv2.imwrite(str(rubber_abs), res["polar_enhanced"])
        cv2.imwrite(str(mask_abs), res["valid_mask"])
        cv2.imwrite(
            str(roi_abs),
            res["roi"],
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )

        return {
            **row_dict,
            "rubber_sheet_path": rubber_rel.as_posix(),
            "mask_path": mask_rel.as_posix(),
            "roi_path": roi_rel.as_posix(),
            "pupil_x": round(seg.pupil_x, 3),
            "pupil_y": round(seg.pupil_y, 3),
            "pupil_r": round(seg.pupil_r, 3),
            "iris_x": round(seg.iris_x, 3),
            "iris_y": round(seg.iris_y, 3),
            "iris_r": round(seg.iris_r, 3),
            "pupil_score": round(seg.pupil_score, 4),
            "iris_score": round(seg.iris_score, 4),
            "seg_method": seg.method,
            "valid_pixel_ratio": round(res["valid_pixel_ratio"], 6),
            "seg_success": bool(res["seg_success"]),
            "prep_error": "",
        }
    except Exception as exc:  # noqa: BLE001 - record failure per file without aborting batch
        return {
            **row_dict,
            "rubber_sheet_path": "",
            "mask_path": "",
            "roi_path": "",
            "pupil_x": float("nan"),
            "pupil_y": float("nan"),
            "pupil_r": float("nan"),
            "iris_x": float("nan"),
            "iris_y": float("nan"),
            "iris_r": float("nan"),
            "pupil_score": float("nan"),
            "iris_score": float("nan"),
            "seg_method": "error",
            "valid_pixel_ratio": 0.0,
            "seg_success": False,
            "prep_error": str(exc),
        }


def _series_stats(series: pd.Series) -> dict[str, float]:
    """Compute concise summary statistics for a numeric Series."""
    clean = series.dropna()
    if clean.empty:
        return {}
    return {
        "mean": round(float(clean.mean()), 4),
        "std": round(float(clean.std()), 4),
        "min": round(float(clean.min()), 4),
        "median": round(float(clean.median()), 4),
        "max": round(float(clean.max()), 4),
    }


def preprocess_dataset(
    metadata_csv: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
    cfg: dict[str, Any] | None = None,
    num_workers: int = 2,
    limit: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Batch-preprocess images listed in metadata_csv and save outputs + summary JSON."""
    merged_cfg = merge_preprocessing_config(cfg)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_csv(metadata_csv, dtype={"subject_id": str})
    if "is_valid" in meta_df.columns:
        meta_df = meta_df[meta_df["is_valid"].astype(bool)].copy()
    if limit is not None and limit > 0:
        meta_df = meta_df.iloc[: int(limit)].copy()

    records = meta_df.to_dict(orient="records")
    tasks = [(rec, str(raw_dir), str(out_path), merged_cfg) for rec in records]

    results: list[dict[str, Any]] = []
    if num_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(
            max_workers=int(num_workers),
            initializer=_init_worker,
        ) as executor:
            iterator = executor.map(_process_record_worker, tasks, chunksize=32)
            if show_progress:
                iterator = tqdm(iterator, total=len(tasks), desc="Preprocessing images")
            for item in iterator:
                results.append(item)
    else:
        iterator = tasks
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Preprocessing images")
        for task in iterator:
            results.append(_process_record_worker(task))

    seg_df = pd.DataFrame(results)
    seg_df.to_csv(out_path / "segmentation_metadata.csv", index=False)

    total = len(seg_df)
    n_success = int(seg_df["seg_success"].sum()) if total > 0 else 0
    summary = {
        "total_processed": int(total),
        "seg_success_count": n_success,
        "seg_success_rate": round(float(n_success / total), 6) if total > 0 else 0.0,
        "method_counts": {
            str(k): int(v) for k, v in seg_df["seg_method"].value_counts().to_dict().items()
        }
        if total > 0
        else {},
        "pupil_r_stats": _series_stats(seg_df["pupil_r"]) if total > 0 else {},
        "iris_r_stats": _series_stats(seg_df["iris_r"]) if total > 0 else {},
        "valid_pixel_ratio_stats": _series_stats(seg_df["valid_pixel_ratio"]) if total > 0 else {},
        "config": merged_cfg,
    }
    with open(out_path / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return seg_df


def archive_processed_dir(processed_dir: str | Path, zip_path: str | Path) -> Path:
    """Archive preprocessed outputs into a single zip file for portable Colab/Drive storage."""
    src_root = Path(processed_dir)
    dst_zip = Path(zip_path)
    dst_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(src_root.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(src_root).as_posix())

    return dst_zip
