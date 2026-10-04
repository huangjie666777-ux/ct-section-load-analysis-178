"""Validation of uploaded NPZ data and reconstruction parameters."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import numpy as np

MAX_ANGLES = 360
MAX_DETECTORS = 512
MAX_OUTPUT_SIZE = 256
# Bound on the *uncompressed* payload to defeat zip-bomb style uploads.
MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024


class ValidationError(ValueError):
    """Raised when an upload or parameter set fails validation."""


@dataclass(frozen=True)
class LoadedData:
    intensity: np.ndarray  # shape (n_angles, n_detectors), float64
    dark: np.ndarray       # shape (n_detectors,), float64
    flat: np.ndarray       # shape (n_detectors,), float64


def _extract_array(archive: np.NpzFile, name: str) -> np.ndarray:
    if name not in archive.files:
        raise ValidationError(f"NPZ is missing required array '{name}'")
    arr = archive[name]
    if arr.dtype.kind == "O":
        raise ValidationError(f"array '{name}' must not be an object array")
    if arr.dtype.kind not in "iuf":
        raise ValidationError(f"array '{name}' must contain numeric data")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _require_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"array '{name}' contains non-finite values")


def load_npz(payload: bytes) -> LoadedData:
    """Load and structurally validate an NPZ upload.

    The archive must contain ``intensity`` (angles x detectors), plus 1-D
    ``dark`` and ``flat`` arrays matching the detector width.
    """
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise ValidationError("uploaded file is not a valid NPZ/ZIP archive")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        total = sum(info.file_size for info in zf.infolist())
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValidationError(
                f"uncompressed payload {total} bytes exceeds limit "
                f"{MAX_UNCOMPRESSED_BYTES} bytes"
            )

    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            intensity = _extract_array(archive, "intensity")
            dark = _extract_array(archive, "dark")
            flat = _extract_array(archive, "flat")
    except zipfile.BadZipFile as exc:
        raise ValidationError("uploaded file is not a valid NPZ archive") from exc
    # np.load with allow_pickle=False raises ValueError on object pickles.
    except ValueError as exc:
        raise ValidationError(f"invalid NPZ contents: {exc}") from exc

    if intensity.ndim != 2:
        raise ValidationError("'intensity' must be a 2-D array")
    n_angles, n_det = intensity.shape
    if not (2 <= n_angles <= MAX_ANGLES):
        raise ValidationError(
            f"number of angles must be in [2, {MAX_ANGLES}], got {n_angles}"
        )
    if not (2 <= n_det <= MAX_DETECTORS):
        raise ValidationError(
            f"detector count must be in [2, {MAX_DETECTORS}], got {n_det}"
        )
    if dark.shape != (n_det,):
        raise ValidationError(
            f"'dark' must have shape ({n_det},), got {dark.shape}"
        )
    if flat.shape != (n_det,):
        raise ValidationError(
            f"'flat' must have shape ({n_det},), got {flat.shape}"
        )

    for arr, name in ((intensity, "intensity"), (dark, "dark"), (flat, "flat")):
        _require_finite(arr, name)

    return LoadedData(intensity=intensity, dark=dark, flat=flat)


def load_mask_npz(payload: bytes, expected_shape: tuple[int, int]) -> np.ndarray:
    """Load a boolean section mask NPZ ('mask' array) matching the image shape."""
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise ValidationError("uploaded mask file is not a valid NPZ/ZIP archive")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        total = sum(info.file_size for info in zf.infolist())
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValidationError(
                f"uncompressed mask payload {total} bytes exceeds limit "
                f"{MAX_UNCOMPRESSED_BYTES} bytes"
            )

    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if "mask" not in archive.files:
                raise ValidationError("mask NPZ is missing required array 'mask'")
            mask = archive["mask"]
    except zipfile.BadZipFile as exc:
        raise ValidationError("uploaded mask file is not a valid NPZ archive") from exc
    except ValueError as exc:
        raise ValidationError(f"invalid mask NPZ contents: {exc}") from exc

    if mask.dtype != np.bool_:
        raise ValidationError(f"'mask' must be a boolean array, got dtype {mask.dtype}")
    if mask.shape != expected_shape:
        raise ValidationError(
            f"'mask' must have shape {expected_shape}, got {mask.shape}"
        )
    if not np.any(mask):
        raise ValidationError("section mask selects no pixels")
    return np.ascontiguousarray(mask)


def validate_params(
    detector_spacing: float,
    center_index: float,
    output_size: int,
    pixel_spacing: float,
    filter_name: str,
) -> dict:
    """Validate scalar reconstruction parameters and return cleaned values."""
    from .reconstruct import supported_filters

    if isinstance(output_size, bool) or not isinstance(output_size, int):
        raise ValidationError("output_size must be an integer")
    if not (1 <= output_size <= MAX_OUTPUT_SIZE):
        raise ValidationError(
            f"output_size must be in [1, {MAX_OUTPUT_SIZE}], got {output_size}"
        )
    if not np.isfinite(detector_spacing) or detector_spacing <= 0:
        raise ValidationError("detector_spacing must be a finite positive number")
    if not np.isfinite(pixel_spacing) or pixel_spacing <= 0:
        raise ValidationError("pixel_spacing must be a finite positive number")
    if not np.isfinite(center_index):
        raise ValidationError("center_index must be finite")
    if filter_name not in supported_filters():
        raise ValidationError(
            f"filter must be one of {sorted(supported_filters())}, got {filter_name!r}"
        )
    return {
        "detector_spacing_mm": float(detector_spacing),
        "center_index": float(center_index),
        "output_size": int(output_size),
        "pixel_spacing_mm": float(pixel_spacing),
        "filter": filter_name,
    }
