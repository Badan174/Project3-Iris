"""Iris preprocessing package: segmentation, Rubber Sheet normalization, and batch pipeline."""

from .iris_segmenter import (
    DEFAULT_PREPROCESSING_CONFIG,
    SegmentationResult,
    detect_and_inpaint_specular,
    detect_iris,
    detect_pupil,
    merge_preprocessing_config,
    refine_circle_ido,
    segment_iris,
)
from .normalization import (
    draw_segmentation_overlay,
    extract_iris_roi,
    normalize_rubber_sheet,
)
from .pipeline import (
    archive_processed_dir,
    preprocess_dataset,
    preprocess_single_image,
)

__all__ = [
    "DEFAULT_PREPROCESSING_CONFIG",
    "SegmentationResult",
    "archive_processed_dir",
    "detect_and_inpaint_specular",
    "detect_iris",
    "detect_pupil",
    "draw_segmentation_overlay",
    "extract_iris_roi",
    "merge_preprocessing_config",
    "normalize_rubber_sheet",
    "preprocess_dataset",
    "preprocess_single_image",
    "refine_circle_ido",
    "segment_iris",
]
