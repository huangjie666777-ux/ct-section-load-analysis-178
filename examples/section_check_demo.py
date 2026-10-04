"""Generate inputs for an eccentric two-material composite-section check.

Phantom: an aluminum disk off-centre at (+6, -4) mm and a plastic disk at
(-8, +5) mm, inside a circular section mask. The script writes the low/high
energy NPZ scans and the boolean mask NPZ, then prints a ready-to-use curl
command for POST /section_check.

Run: .venv/bin/python examples/section_check_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctrecon.synthetic import acquire, disk_sinogram

OUT_DIR = Path(__file__).resolve().parent

GEOMETRY = dict(
    n_angles=180,
    n_det=256,
    detector_spacing_mm=0.5,
    center_index=127.5,
    output_size=128,
    pixel_spacing_mm=0.5,
)

MATERIALS = [
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
MU_MATRIX = [[0.06, 0.02], [0.03, 0.018]]
DISKS = [
    ((6.0, -4.0), 12.0, 2.7),    # aluminum
    ((-8.0, 5.0), 9.0, 1.2),     # plastic
]
LOAD_CASES = [
    {"name": "service", "axial_force_n": 20000.0, "mx_nmm": 150000.0, "my_nmm": -60000.0},
    {"name": "overload", "axial_force_n": -50000.0, "mx_nmm": -300000.0, "my_nmm": 120000.0},
]


def main() -> None:
    g = GEOMETRY
    material_sinograms = [
        disk_sinogram(
            n_angles=g["n_angles"],
            n_det=g["n_det"],
            detector_spacing=g["detector_spacing_mm"],
            center_index=g["center_index"],
            disk_center_mm=center,
            radius_mm=radius,
            attenuation=density,
        )
        for center, radius, density in DISKS
    ]
    mu = np.asarray(MU_MATRIX)
    line_integrals = np.einsum("em,mad->ead", mu, np.stack(material_sinograms))

    rng = np.random.default_rng(0)
    for label, integrals in zip(("low", "high"), line_integrals):
        intensity, dark, flat = acquire(integrals, rng=rng)
        path = OUT_DIR / ("section_check_" + label + ".npz")
        np.savez(path, intensity=intensity, dark=dark, flat=flat)
        print("wrote", path)

    n = g["output_size"]
    yy, xx = np.mgrid[0:n, 0:n]
    centre = (n - 1) / 2.0
    mask = (xx - centre) ** 2 + (yy - centre) ** 2 <= (n * 0.45) ** 2
    mask_path = OUT_DIR / "section_check_mask.npz"
    np.savez(mask_path, mask=mask)
    print("wrote", mask_path)

    lines = [
        "curl -s -X POST http://127.0.0.1:8000/section_check",
        "  -F 'low_file=@examples/section_check_low.npz'",
        "  -F 'high_file=@examples/section_check_high.npz'",
        "  -F 'mask_file=@examples/section_check_mask.npz'",
        "  -F 'detector_spacing_mm=" + str(g["detector_spacing_mm"]) + "'",
        "  -F 'center_index=" + str(g["center_index"]) + "'",
        "  -F 'output_size=" + str(g["output_size"]) + "'",
        "  -F 'pixel_spacing_mm=" + str(g["pixel_spacing_mm"]) + "'",
        "  -F 'filter=hann'",
        "  -F 'materials=" + json.dumps(MATERIALS) + "'",
        "  -F 'mu_matrix=" + json.dumps(MU_MATRIX) + "'",
        "  -F 'load_cases=" + json.dumps(LOAD_CASES) + "'",
        "  -o section_check.zip",
    ]
    print((" " + chr(92) + chr(10)).join(lines))


if __name__ == "__main__":
    main()
