"""FastAPI application exposing the parallel-beam CT reconstruction API."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from .dual_service import decompose_densities, dual_energy_pipeline
from .io_utils import ValidationError, load_mask_npz, load_npz
from .preview import npy_bytes, render_png
from .reconstruct import supported_filters
from .section_check import report_json, run_section_check, validate_section_materials
from .service import reconstruct_upload

app = FastAPI(title="Parallel-beam CT FBP reconstruction", version="1.0.0")


@app.exception_handler(ValidationError)
async def _validation_handler(_request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "filters": list(supported_filters())}


@app.post("/reconstruct")
async def reconstruct(
    file: UploadFile = File(..., description="NPZ with intensity, dark, flat"),
    detector_spacing_mm: float = Form(..., alias="detector_spacing_mm"),
    center_index: float = Form(...),
    output_size: int = Form(...),
    pixel_spacing_mm: float = Form(...),
    filter: str = Form("ram-lak"),
) -> Response:
    payload = await file.read()
    data = load_npz(payload)
    result = reconstruct_upload(
        data,
        {
            "detector_spacing_mm": detector_spacing_mm,
            "center_index": center_index,
            "output_size": output_size,
            "pixel_spacing_mm": pixel_spacing_mm,
            "filter": filter,
        },
    )

    metadata = {
        "parameters": result.params,
        "ranges": {
            "image_min": result.stats["image_min"],
            "image_max": result.stats["image_max"],
            "image_mean": result.stats["image_mean"],
            "sinogram_min": result.stats["sinogram_min"],
            "sinogram_max": result.stats["sinogram_max"],
        },
        "units": {
            "detector_spacing_mm": "millimeter",
            "pixel_spacing_mm": "millimeter",
            "image": "linear attenuation coefficient per millimeter",
        },
        "notes": "preview.png uses a min/max linear stretch for display only; reconstruction.npy is untouched float64 data.",
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("reconstruction.npy", npy_bytes(result.image))
        zf.writestr("preview.png", render_png(result.image))
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, sort_keys=True))

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="reconstruction.zip"'},
    )


@app.post("/decompose")
async def decompose(
    low_file: UploadFile = File(..., description="NPZ with low-energy intensity, dark, flat"),
    high_file: UploadFile = File(..., description="NPZ with high-energy intensity, dark, flat"),
    detector_spacing_mm: float = Form(...),
    center_index: float = Form(...),
    output_size: int = Form(...),
    pixel_spacing_mm: float = Form(...),
    filter: str = Form("ram-lak"),
    materials: str = Form(..., description='JSON list of two material names'),
    mu_matrix: str = Form(..., description="JSON 2x2 mass attenuation matrix, mm^2/mg"),
    slice_thickness_mm: float = Form(...),
    rois: str = Form(..., description="JSON list of ROI rectangles"),
) -> Response:
    low_data = load_npz(await low_file.read())
    high_data = load_npz(await high_file.read())
    result = dual_energy_pipeline(
        low_data,
        high_data,
        {
            "detector_spacing_mm": detector_spacing_mm,
            "center_index": center_index,
            "output_size": output_size,
            "pixel_spacing_mm": pixel_spacing_mm,
            "filter": filter,
        },
        materials,
        mu_matrix,
        slice_thickness_mm,
        rois,
    )

    metadata = {
        "parameters": result.params,
        "rois": result.roi_results,
        "ranges": {
            f"density_{name}": {
                "min": float(np.min(density)),
                "max": float(np.max(density)),
            }
            for name, density in result.densities.items()
        },
        "units": {
            "density": "mg/mm^3",
            "residual": "linear attenuation per millimeter (mm^-1)",
            "mass": "milligram",
            "mu_matrix": "mm^2/mg",
        },
        "notes": "preview PNGs use a min/max linear stretch for display only; "
        "density NPY files are untouched float64 data.",
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, density in result.densities.items():
            zf.writestr(f"density_{name}.npy", npy_bytes(density))
        for energy, residual in result.residuals.items():
            zf.writestr(f"residual_{energy}.npy", npy_bytes(residual))
        for name, density in result.densities.items():
            zf.writestr(f"preview_{name}.png", render_png(density))
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, sort_keys=True))

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="decomposition.zip"'},
    )


@app.post("/section_check")
async def section_check(
    low_file: UploadFile = File(..., description="NPZ with low-energy intensity, dark, flat"),
    high_file: UploadFile = File(..., description="NPZ with high-energy intensity, dark, flat"),
    mask_file: UploadFile = File(..., description="NPZ with boolean 'mask' array, image-sized"),
    detector_spacing_mm: float = Form(...),
    center_index: float = Form(...),
    output_size: int = Form(...),
    pixel_spacing_mm: float = Form(...),
    filter: str = Form("ram-lak"),
    materials: str = Form(..., description="JSON list of two material objects with mechanical data"),
    mu_matrix: str = Form(..., description="JSON 2x2 mass attenuation matrix, mm^2/mg"),
    load_cases: str = Form(..., description="JSON list of 1-8 load cases with N, Mx, My"),
) -> Response:
    low_data = load_npz(await low_file.read())
    high_data = load_npz(await high_file.read())
    try:
        materials_spec = json.loads(materials)
    except json.JSONDecodeError as exc:
        raise ValidationError("field 'materials' is not valid JSON") from exc
    try:
        load_cases_spec = json.loads(load_cases)
    except json.JSONDecodeError as exc:
        raise ValidationError("field 'load_cases' is not valid JSON") from exc
    material_specs = validate_section_materials(materials_spec)

    geometry = {
        "detector_spacing_mm": detector_spacing_mm,
        "center_index": center_index,
        "output_size": output_size,
        "pixel_spacing_mm": pixel_spacing_mm,
        "filter": filter,
    }
    densities, params, _ = decompose_densities(
        low_data,
        high_data,
        geometry,
        json.dumps([m.name for m in material_specs]),
        mu_matrix,
    )
    mask = load_mask_npz(await mask_file.read(), (params["output_size"],) * 2)

    result = run_section_check(
        densities,
        mask,
        pixel_spacing_mm=params["pixel_spacing_mm"],
        materials_raw=materials_spec,
        load_cases_raw=load_cases_spec,
    )

    report = report_json(result)
    report["parameters"] = params

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for case in result.load_cases:
            for material in result.materials:
                zf.writestr(
                    f"stress_{case.name}_{material.name}.npy",
                    npy_bytes(result.stress_maps[case.name][material.name]),
                )
            zf.writestr(
                f"exceedance_{case.name}.png",
                render_png(result.utilization_maps[case.name]),
            )
        zf.writestr("report.json", json.dumps(report, indent=2, sort_keys=True))

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="section_check.zip"'},
    )
