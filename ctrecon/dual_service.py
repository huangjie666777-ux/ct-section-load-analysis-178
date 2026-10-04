"""Dual-energy pipeline: two aligned scans -> material densities + ROI masses."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .calibration import calibrate
from .decompose import decompose, validate_materials, validate_mu_matrix
from .io_utils import LoadedData, ValidationError, validate_params
from .reconstruct import fbp
from .roi import integrate_rois, validate_rois, validate_slice_thickness


@dataclass(frozen=True)
class DualEnergyResult:
    densities: dict[str, np.ndarray]      # material name -> mg/mm^3 map
    residuals: dict[str, np.ndarray]      # 'low'/'high' -> mm^-1 residual map
    attenuation: dict[str, np.ndarray]    # 'low'/'high' -> mm^-1 FBP map
    params: dict
    roi_results: list[dict]


def _parse_json_field(raw: str, name: str) -> object:
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"field {name!r} is not valid JSON") from exc


def decompose_densities(
    low_data: LoadedData,
    high_data: LoadedData,
    geometry: dict,
    materials_raw: str,
    mu_matrix_raw: str,
) -> tuple[dict[str, np.ndarray], dict, dict[str, np.ndarray]]:
    """Shared reconstruction + decomposition for /decompose and /section_check.

    Returns (densities, params, attenuation_maps); density maps are the
    untouched float64 outputs of the per-pixel NNLS decomposition.
    """
    if low_data.intensity.shape != high_data.intensity.shape:
        raise ValidationError(
            "low and high energy intensity arrays must have identical shapes, "
            f"got {low_data.intensity.shape} and {high_data.intensity.shape}"
        )

    clean = validate_params(
        geometry["detector_spacing_mm"],
        geometry["center_index"],
        geometry["output_size"],
        geometry["pixel_spacing_mm"],
        geometry["filter"],
    )
    materials = validate_materials(_parse_json_field(materials_raw, "materials"))
    matrix = validate_mu_matrix(_parse_json_field(mu_matrix_raw, "mu_matrix"))

    maps = {}
    for label, data in (("low", low_data), ("high", high_data)):
        sinogram = calibrate(data)
        maps[label] = fbp(
            sinogram,
            detector_spacing=clean["detector_spacing_mm"],
            center_index=clean["center_index"],
            output_size=clean["output_size"],
            pixel_spacing=clean["pixel_spacing_mm"],
            filter_name=clean["filter"],
        )

    rho1, rho2, res_low, res_high = decompose(matrix, maps["low"], maps["high"])
    densities = {materials[0]: rho1, materials[1]: rho2}
    params = {
        **clean,
        "materials": list(materials),
        "mu_matrix_mm2_per_mg": matrix.tolist(),
    }
    attenuation = dict(maps)
    attenuation["residual_low"] = res_low
    attenuation["residual_high"] = res_high
    return densities, params, attenuation


def dual_energy_pipeline(
    low_data: LoadedData,
    high_data: LoadedData,
    geometry: dict,
    materials_raw: str,
    mu_matrix_raw: str,
    slice_thickness_mm: object,
    rois_raw: str,
) -> DualEnergyResult:
    densities, params, attenuation = decompose_densities(
        low_data, high_data, geometry, materials_raw, mu_matrix_raw
    )
    thickness = validate_slice_thickness(slice_thickness_mm)
    rois = validate_rois(
        _parse_json_field(rois_raw, "rois"), params["output_size"]
    )
    residuals = {
        "low": attenuation.pop("residual_low"),
        "high": attenuation.pop("residual_high"),
    }
    roi_results = integrate_rois(
        rois,
        densities,
        residuals,
        pixel_spacing_mm=params["pixel_spacing_mm"],
        slice_thickness_mm=thickness,
    )
    params = {**params, "slice_thickness_mm": thickness}
    return DualEnergyResult(
        densities=densities,
        residuals=residuals,
        attenuation=attenuation,
        params=params,
        roi_results=roi_results,
    )
