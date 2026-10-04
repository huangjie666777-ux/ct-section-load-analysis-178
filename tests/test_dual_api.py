import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ctrecon.app import app
from ctrecon.synthetic import acquire, disk_sinogram

client = TestClient(app)
PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")

GEOMETRY = dict(
    detector_spacing_mm="0.5",
    center_index="127.5",
    output_size="128",
    pixel_spacing_mm="0.5",
    filter="hann",
    slice_thickness_mm="1.0",
)
MATERIALS = ["aluminum", "plastic"]
MU = [[0.06, 0.02], [0.03, 0.018]]
DISKS = [((6.0, -4.0), 12.0, 2.7), ((-8.0, 5.0), 9.0, 1.2)]
ROIS = [
    {"name": "al_disk", "x0": 48, "y0": 44, "x1": 102, "y1": 98},
    {"name": "plastic_disk", "x0": 26, "y0": 32, "x1": 68, "y1": 82},
]


def _npz(intensity, dark, flat):
    buffer = io.BytesIO()
    np.savez(buffer, intensity=intensity, dark=dark, flat=flat)
    return buffer.getvalue()


def _phantom_npzs():
    sino = [
        disk_sinogram(180, 256, 0.5, 127.5, center, radius, density)
        for center, radius, density in DISKS
    ]
    mu = np.asarray(MU)
    combined = np.einsum("em,mad->ead", mu, np.stack(sino))
    files = []
    for integrals in combined:
        intensity, dark, flat = acquire(integrals)
        files.append(_npz(intensity, dark, flat))
    return files


def _post(low, high, **overrides):
    params = dict(GEOMETRY)
    params.update(
        materials=json.dumps(MATERIALS),
        mu_matrix=json.dumps(MU),
        rois=json.dumps(ROIS),
    )
    params.update(overrides)
    files = {
        "low_file": ("low.npz", low, "application/octet-stream"),
        "high_file": ("high.npz", high, "application/octet-stream"),
    }
    return client.post("/decompose", files=files, data=params)


def test_decompose_endpoint_bundle_and_quantities():
    low, high = _phantom_npzs()
    response = _post(low, high)
    assert response.status_code == 200
    bundle = zipfile.ZipFile(io.BytesIO(response.content))
    assert set(bundle.namelist()) == {
        "density_aluminum.npy",
        "density_plastic.npy",
        "residual_low.npy",
        "residual_high.npy",
        "preview_aluminum.png",
        "preview_plastic.png",
        "metadata.json",
    }
    assert bundle.read("preview_aluminum.png")[:8] == PNG_MAGIC

    rho_al = np.load(io.BytesIO(bundle.read("density_aluminum.npy")), allow_pickle=False)
    assert rho_al.dtype == np.float64
    assert rho_al.shape == (128, 128)
    assert rho_al.min() >= 0.0

    metadata = json.loads(bundle.read("metadata.json"))
    assert metadata["parameters"]["materials"] == MATERIALS
    by_name = {roi["name"]: roi for roi in metadata["rois"]}
    # Expected mass: density * pi R^2 * thickness (ROI covers the full disk).
    al_expected = 2.7 * np.pi * 12.0**2 * 1.0
    plastic_expected = 1.2 * np.pi * 9.0**2 * 1.0
    assert by_name["al_disk"]["mass_mg"]["aluminum"] == pytest.approx(
        al_expected, rel=0.1
    )
    assert by_name["plastic_disk"]["mass_mg"]["plastic"] == pytest.approx(
        plastic_expected, rel=0.1
    )
    assert abs(by_name["al_disk"]["mean_residual_per_mm"]["low"]) < 0.05


def test_decompose_rejects_shape_mismatch():
    low, _ = _phantom_npzs()
    intensity, dark, flat = acquire(np.zeros((90, 128)))
    response = _post(low, _npz(intensity, dark, flat),
                     center_index="63.5")
    assert response.status_code == 422


def test_decompose_rejects_bad_inputs():
    low, high = _phantom_npzs()
    bad_cases = [
        {"materials": json.dumps(["a", "a"])},
        {"materials": "not json"},
        {"mu_matrix": json.dumps([[0.1, 0.2], [0.3, -0.1]])},
        {"mu_matrix": json.dumps([[0.1, 0.2], [0.3]])},
        {"mu_matrix": json.dumps([[1.0, 1.0], [1.0, 1.0000001]])},
        {"slice_thickness_mm": "0"},
        {"rois": json.dumps([{"name": "x", "x0": 0, "y0": 0, "x1": 999, "y1": 2}])},
        {"rois": json.dumps([])},
    ]
    for overrides in bad_cases:
        response = _post(low, high, **overrides)
        assert response.status_code == 422, overrides
        assert "error" in response.json()
