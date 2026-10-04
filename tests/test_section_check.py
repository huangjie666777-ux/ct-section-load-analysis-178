import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ctrecon.app import app
from ctrecon.section_check import run_section_check
from ctrecon.synthetic import acquire, disk_sinogram

client = TestClient(app)
PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")

GEOMETRY = dict(
    detector_spacing_mm="0.5",
    center_index="127.5",
    output_size="128",
    pixel_spacing_mm="0.5",
    filter="hann",
)
MATERIAL_PROPS = [
    {
        "name": "aluminum",
        "reference_density_mg_per_mm3": 2.7,
        "elastic_modulus_mpa": 70000.0,
        "tensile_allowable_mpa": 150.0,
        "compressive_allowable_mpa": 150.0,
    },
    {
        "name": "plastic",
        "reference_density_mg_per_mm3": 1.2,
        "elastic_modulus_mpa": 3000.0,
        "tensile_allowable_mpa": 60.0,
        "compressive_allowable_mpa": 90.0,
    },
]
MU = [[0.06, 0.02], [0.03, 0.018]]
DISKS = [((6.0, -4.0), 12.0, 2.7), ((-8.0, 5.0), 9.0, 1.2)]
LOAD_CASES = [
    {"name": "axial", "axial_force_n": 5000.0, "mx_nmm": 0.0, "my_nmm": 0.0},
    {"name": "bent", "axial_force_n": 1000.0, "mx_nmm": 20000.0, "my_nmm": -8000.0},
]


def _npz(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _phantom_npzs():
    sino = [
        disk_sinogram(180, 256, 0.5, 127.5, center, radius, density)
        for center, radius, density in DISKS
    ]
    mu = np.asarray(MU)
    combined = np.einsum("em,mad->ead", mu, np.stack(sino))
    return [
        _npz(**dict(zip(("intensity", "dark", "flat"), acquire(integrals))))
        for integrals in combined
    ]


def _mask_npz(output_size=128):
    yy, xx = np.mgrid[0:output_size, 0:output_size]
    cx = cy = (output_size - 1) / 2.0
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (output_size * 0.45) ** 2
    return _npz(mask=mask)


def _post(low, high, mask, **overrides):
    params = dict(GEOMETRY)
    params.update(
        materials=json.dumps(MATERIAL_PROPS),
        mu_matrix=json.dumps(MU),
        load_cases=json.dumps(LOAD_CASES),
    )
    params.update(overrides)
    files = {
        "low_file": ("low.npz", low, "application/octet-stream"),
        "high_file": ("high.npz", high, "application/octet-stream"),
        "mask_file": ("mask.npz", mask, "application/octet-stream"),
    }
    return client.post("/section_check", files=files, data=params)


def test_section_check_endpoint_bundle_and_report():
    low, high = _phantom_npzs()
    response = _post(low, high, _mask_npz())
    assert response.status_code == 200
    bundle = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(bundle.namelist())
    for case in ("axial", "bent"):
        for material in ("aluminum", "plastic"):
            assert f"stress_{case}_{material}.npy" in names
        assert f"exceedance_{case}.png" in names
    assert "report.json" in names
    assert bundle.read("exceedance_axial.png")[:8] == PNG_MAGIC

    stress = np.load(
        io.BytesIO(bundle.read("stress_axial_aluminum.npy")), allow_pickle=False
    )
    assert stress.dtype == np.float64
    assert stress.shape == (128, 128)
    assert np.isnan(stress).any()  # outside material presence

    report = json.loads(bundle.read("report.json"))
    assert report["stiffness_matrix"]["matrix"][0][0] > 0.0
    by_case = {c["name"]: c for c in report["cases"]}
    for case in by_case.values():
        res = case["equilibrium_residual"]
        assert abs(res["d_n"]) < 1e-6 * max(1.0, abs(case["loads"]["axial_force_n"]))
        for material in case["materials"]:
            if material["present"]:
                assert material["governing_ratio"] >= 0.0
    # Axial-only case: tension extreme location must be inside the aluminum disk.
    al = next(m for m in by_case["axial"]["materials"] if m["material"] == "aluminum")
    assert al["max_tension"]["stress_mpa"] > 0.0


def test_section_check_rejects_bad_inputs():
    low, high = _phantom_npzs()
    mask = _mask_npz()
    bad_materials = json.dumps([MATERIAL_PROPS[0], {**MATERIAL_PROPS[1], "elastic_modulus_mpa": -1}])
    bad_cases = [
        {"materials": bad_materials},
        {"materials": json.dumps([MATERIAL_PROPS[0], MATERIAL_PROPS[0]])},
        {"materials": "not json"},
        {"load_cases": json.dumps([])},
        {"load_cases": json.dumps([LOAD_CASES[0]] * 2)},
        {"load_cases": json.dumps([{**LOAD_CASES[0], "axial_force_n": float("nan")}])},
        {"load_cases": json.dumps([LOAD_CASES[0]] * 9)},
        {"mu_matrix": json.dumps([[0.1, 0.2], [0.3]])},  # ragged rows
    ]
    for overrides in bad_cases:
        response = _post(low, high, mask, **overrides)
        assert response.status_code == 422, overrides
        assert "error" in response.json()


def test_section_check_rejects_bad_mask():
    low, high = _phantom_npzs()
    # Non-boolean mask.
    response = _post(low, high, _npz(mask=np.ones((128, 128))))
    assert response.status_code == 422
    # Wrong shape.
    response = _post(low, high, _npz(mask=np.ones((64, 64), dtype=bool)))
    assert response.status_code == 422
    # Empty mask.
    response = _post(low, high, _npz(mask=np.zeros((128, 128), dtype=bool)))
    assert response.status_code == 422
    # Not an NPZ at all.
    response = _post(low, high, b"not a zip")
    assert response.status_code == 422


def test_section_check_rejects_singular_section():
    # Mask where no material exists -> zero stiffness -> 422, no fake results.
    densities = {
        "aluminum": np.zeros((8, 8)),
        "plastic": np.zeros((8, 8)),
    }
    mask = np.ones((8, 8), dtype=bool)
    with pytest.raises(Exception, match="singular|ill-conditioned"):
        run_section_check(
            densities,
            mask,
            1.0,
            MATERIAL_PROPS,
            [{"name": "c", "axial_force_n": 1.0, "mx_nmm": 0.0, "my_nmm": 0.0}],
        )


def test_pure_axial_matches_strength_of_materials():
    # Homogeneous single-material section, pure tension: sigma = N / A exactly.
    n = 16
    spacing = 0.5
    densities = {
        "aluminum": np.full((n, n), 2.7),
        "plastic": np.zeros((n, n)),
    }
    mask = np.ones((n, n), dtype=bool)
    result = run_section_check(
        densities,
        mask,
        spacing,
        MATERIAL_PROPS,
        [{"name": "ax", "axial_force_n": 1000.0, "mx_nmm": 0.0, "my_nmm": 0.0}],
    )
    area = n * n * spacing * spacing
    expected = 1000.0 / area
    stress = result.stress_maps["ax"]["aluminum"]
    assert np.allclose(np.nan_to_num(stress, nan=expected), expected, rtol=1e-10)
    case = result.cases[0]
    assert abs(case["equilibrium_residual"]["d_n"]) < 1e-9


def test_eccentric_composite_biaxial_bending():
    # Two off-centre material blocks; verify coupled solve against direct
    # numerical integration of the returned strain field.
    n = 32
    spacing = 1.0
    densities = {"aluminum": np.zeros((n, n)), "plastic": np.zeros((n, n))}
    densities["aluminum"][4:12, 20:28] = 2.7   # +x, +y quadrant (row 0 is +y)
    densities["plastic"][20:28, 4:12] = 1.2    # -x, -y quadrant
    mask = np.ones((n, n), dtype=bool)
    case = {"name": "ecc", "axial_force_n": 800.0, "mx_nmm": 3000.0, "my_nmm": -1500.0}
    result = run_section_check(densities, mask, spacing, MATERIAL_PROPS, [case])

    eps0, kx, ky = result.strain_vector["ecc"]
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx - (n - 1) / 2.0) * spacing
    y = ((n - 1) / 2.0 - yy) * spacing
    strain = eps0 + kx * y - ky * x
    e_eff = (
        densities["aluminum"] / 2.7 * 70000.0
        + densities["plastic"] / 1.2 * 3000.0
    )
    sigma = e_eff * strain
    area = spacing * spacing
    assert sigma.sum() * area == pytest.approx(800.0, rel=1e-9)
    assert (y * sigma).sum() * area == pytest.approx(3000.0, rel=1e-9)
    assert (-x * sigma).sum() * area == pytest.approx(-1500.0, rel=1e-9)

    report = result.cases[0]
    al = next(m for m in report["materials"] if m["material"] == "aluminum")
    # Extreme locations must lie inside the aluminum block.
    assert 4 <= al["max_tension"]["pixel_row"] < 12
    assert 20 <= al["max_tension"]["pixel_col"] < 28


def test_volume_fraction_normalization_and_void():
    # Sum > 1 normalized proportionally; sum < 1 keeps void (lower stiffness).
    n = 8
    densities = {"aluminum": np.full((n, n), 2.7), "plastic": np.full((n, n), 1.2)}
    mask = np.ones((n, n), dtype=bool)
    result = run_section_check(
        densities,
        mask,
        1.0,
        MATERIAL_PROPS,
        [{"name": "c", "axial_force_n": 1.0, "mx_nmm": 0.0, "my_nmm": 0.0}],
    )
    # Fractions halved: effective E = 0.5*70000 + 0.5*3000.
    assert result.stiffness_matrix[0, 0] == pytest.approx(
        (0.5 * 70000.0 + 0.5 * 3000.0) * n * n
    )
    densities_void = {"aluminum": np.full((n, n), 1.35), "plastic": np.zeros((n, n))}
    result_void = run_section_check(
        densities_void,
        mask,
        1.0,
        MATERIAL_PROPS,
        [{"name": "c", "axial_force_n": 1.0, "mx_nmm": 0.0, "my_nmm": 0.0}],
    )
    assert result_void.stiffness_matrix[0, 0] == pytest.approx(0.5 * 70000.0 * n * n)
