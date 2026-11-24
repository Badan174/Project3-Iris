"""Dataset scanning, validation, splitting, and leakage verification helpers."""

from .dataset_utils import (
    build_dataset_summary,
    create_closed_set_splits,
    create_open_set_splits,
    generate_verification_pairs,
    scan_casia_thousand,
    validate_images,
    verify_no_leakage,
)

__all__ = [
    "build_dataset_summary",
    "scan_casia_thousand",
    "validate_images",
    "create_closed_set_splits",
    "create_open_set_splits",
    "generate_verification_pairs",
    "verify_no_leakage",
]
