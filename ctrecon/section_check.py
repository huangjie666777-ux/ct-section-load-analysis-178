"""Composite-section load check on dual-energy density maps.

Linear elasticity, perfect bond between materials, plane sections remain
plane. Volume fractions come from partial densities divided by each
material's reference density; fractions summing above 1 are normalized
proportionally, sums below 1 keep the void. Pixels outside the section
mask take no part; the density maps themselves are never modified.

Strain at a pixel centre: eps = eps0 + kx*y - ky*x, with the origin at the
image centre, +x to the right, +y up (array row 0 is +y). Resultants are
N = integral(sigma dA), Mx = integral(y sigma dA), My = integral(-x sigma
dA); the 3x3 coupled system is solved jointly, coupling terms included.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_utils import ValidationError

MAX_LOAD_CASES = 8
MAX_STIFFNESS_CONDITION = 1e12


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    reference_density_mg_per_mm3: float
    elastic_modulus_mpa: float
    tensile_allowable_mpa: float
    compressive_allowable_mpa: float


@dataclass(frozen=True)
class LoadCase:
    name: str
    axial_force_n: float
    mx_nmm: float
    my_nmm: float


@dataclass(frozen=True)
class SectionCheckResult:
    stiffness_matrix: np.ndarray          # 3x3 coupled section stiffness
    strain_vector: dict[str, np.ndarray]  # case name -> [eps0, kx, ky]
    residual: dict[str, np.ndarray]       # case name -> [dN, dMx, dMy]
    stress_maps: dict[str, dict[str, np.ndarray]]  # case -> material -> MPa (NaN outside material)
    utilization_maps: dict[str, np.ndarray]        # case -> max utilization ratio (0 outside)
    cases: list[dict]                     # per-case JSON-able verdicts
    materials: tuple[MaterialSpec, MaterialSpec]
    load_cases: list[LoadCase]
    pixel_spacing_mm: float
    n_section_pixels: int


def _finite_positive(value: object, field: str, material: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"material {material!r}: {field} must be a number")
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValidationError(
            f"material {material!r}: {field} must be finite and strictly positive"
        )
    return out


def validate_section_materials(spec: object) -> tuple[MaterialSpec, MaterialSpec]:
    """Validate two uniquely named materials with positive mechanical data."""
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        raise ValidationError("materials must be a list of two objects")
    keys = (
        "name",
        "reference_density_mg_per_mm3",
        "elastic_modulus_mpa",
        "tensile_allowable_mpa",
        "compressive_allowable_mpa",
    )
    materials: list[MaterialSpec] = []
    for entry in spec:
        if not isinstance(entry, dict):
            raise ValidationError("each material must be an object")
        missing = [k for k in keys if k not in entry]
        if missing:
            raise ValidationError(f"material is missing keys {missing}")
        name = entry["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("each material needs a non-empty name")
        name = name.strip()
        materials.append(
            MaterialSpec(
                name=name,
                reference_density_mg_per_mm3=_finite_positive(
                    entry["reference_density_mg_per_mm3"],
                    "reference_density_mg_per_mm3",
                    name,
                ),
                elastic_modulus_mpa=_finite_positive(
                    entry["elastic_modulus_mpa"], "elastic_modulus_mpa", name
                ),
                tensile_allowable_mpa=_finite_positive(
                    entry["tensile_allowable_mpa"], "tensile_allowable_mpa", name
                ),
                compressive_allowable_mpa=_finite_positive(
                    entry["compressive_allowable_mpa"],
                    "compressive_allowable_mpa",
                    name,
                ),
            )
        )
    if materials[0].name == materials[1].name:
        raise ValidationError("material names must be unique")
    return materials[0], materials[1]


def validate_load_cases(spec: object) -> list[LoadCase]:
    """Validate 1-8 uniquely named load cases with finite N, Mx, My."""
    if not isinstance(spec, (list, tuple)) or not (1 <= len(spec) <= MAX_LOAD_CASES):
        raise ValidationError(
            f"load_cases must be a list of 1 to {MAX_LOAD_CASES} cases"
        )
    cases: list[LoadCase] = []
    names: set[str] = set()
    for entry in spec:
        if not isinstance(entry, dict):
            raise ValidationError("each load case must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("each load case needs a non-empty name")
        name = name.strip()
        if name in names:
            raise ValidationError(f"duplicate load case name {name!r}")
        names.add(name)
        values = []
        for key in ("axial_force_n", "mx_nmm", "my_nmm"):
            if key not in entry:
                raise ValidationError(f"load case {name!r} is missing key {key!r}")
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(f"load case {name!r}: {key} must be a number")
            value = float(value)
            if not np.isfinite(value):
                raise ValidationError(f"load case {name!r}: {key} must be finite")
            values.append(value)
        cases.append(
            LoadCase(
                name=name,
                axial_force_n=values[0],
                mx_nmm=values[1],
                my_nmm=values[2],
            )
        )
    return cases


def pixel_coordinates(
    shape: tuple[int, int], spacing_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates (mm), origin at image centre, +x right, +y up."""
    rows, cols = shape
    col = np.arange(cols, dtype=np.float64)
    row = np.arange(rows, dtype=np.float64)
    x = (col - (cols - 1) / 2.0) * spacing_mm
    y = ((rows - 1) / 2.0 - row) * spacing_mm
    return np.meshgrid(x, y)  # both (rows, cols)


def volume_fractions(
    densities: dict[str, np.ndarray],
    materials: tuple[MaterialSpec, MaterialSpec],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """phi_m = rho_m / rho_ref_m inside the mask; pixels summing above 1 are
    normalized proportionally, sums below 1 keep the void. Zero outside mask."""
    phi = {}
    for material in materials:
        density = densities[material.name]
        if density.shape != mask.shape:
            raise ValidationError(
                f"density map for {material.name!r} has shape {density.shape}, "
                f"expected {mask.shape}"
            )
        phi[material.name] = np.where(
            mask, density / material.reference_density_mg_per_mm3, 0.0
        )
    total = phi[materials[0].name] + phi[materials[1].name]
    over = total > 1.0
    if np.any(over):
        scale = np.where(over, 1.0 / np.where(over, total, 1.0), 1.0)
        for name in phi:
            phi[name] = phi[name] * scale
    return phi


def section_stiffness(
    phi: dict[str, np.ndarray],
    materials: tuple[MaterialSpec, MaterialSpec],
    mask: np.ndarray,
    pixel_spacing_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the coupled 3x3 section stiffness for [eps0, kx, ky].

    Returns (K, x, y, e_eff) with x/y in mm and e_eff the effective modulus
    map (MPa). K maps [eps0, kx(1/mm), ky(1/mm)] to [N, N*mm, N*mm].
    """
    x, y = pixel_coordinates(mask.shape, pixel_spacing_mm)
    e_eff = phi[materials[0].name] * materials[0].elastic_modulus_mpa
    e_eff = e_eff + phi[materials[1].name] * materials[1].elastic_modulus_mpa
    area = pixel_spacing_mm * pixel_spacing_mm
    w = e_eff * area
    s = float(w.sum())
    sx = float((w * x).sum())
    sy = float((w * y).sum())
    sxx = float((w * x * x).sum())
    syy = float((w * y * y).sum())
    sxy = float((w * x * y).sum())
    K = np.array(
        [
            [s, sy, -sx],
            [sy, syy, -sxy],
            [-sx, -sxy, sxx],
        ]
    )
    cond = float(np.linalg.cond(K, 2))
    if not np.isfinite(cond) or cond > MAX_STIFFNESS_CONDITION or s <= 0.0:
        raise ValidationError(
            "section stiffness is singular or ill-conditioned "
            f"(2-norm condition {cond:.6g}); check mask and material data"
        )
    return K, x, y, e_eff


def _extreme(values, xs, ys, rows, cols, pick) -> dict:
    idx = pick(values)
    return {
        "stress_mpa": float(values[idx]),
        "x_mm": float(xs[idx]),
        "y_mm": float(ys[idx]),
        "pixel_row": int(rows[idx]),
        "pixel_col": int(cols[idx]),
    }


def run_section_check(
    densities: dict[str, np.ndarray],
    mask: np.ndarray,
    pixel_spacing_mm: float,
    materials_raw: object,
    load_cases_raw: object,
) -> SectionCheckResult:
    """Full composite-section check for every load case."""
    materials = validate_section_materials(materials_raw)
    load_cases = validate_load_cases(load_cases_raw)
    if mask.shape != next(iter(densities.values())).shape:
        raise ValidationError(
            f"mask shape {mask.shape} does not match density maps "
            f"{next(iter(densities.values())).shape}"
        )
    if not np.any(mask):
        raise ValidationError("section mask selects no pixels")

    phi = volume_fractions(densities, materials, mask)
    K, x, y, _ = section_stiffness(phi, materials, mask, pixel_spacing_mm)

    stress_maps: dict[str, dict[str, np.ndarray]] = {}
    utilization_maps: dict[str, np.ndarray] = {}
    strain_vector: dict[str, np.ndarray] = {}
    residual: dict[str, np.ndarray] = {}
    case_reports: list[dict] = []

    for case in load_cases:
        load = np.array([case.axial_force_n, case.mx_nmm, case.my_nmm])
        try:
            solution = np.linalg.solve(K, load)
        except np.linalg.LinAlgError as exc:
            raise ValidationError("section stiffness is singular; cannot solve") from exc
        eps0, kx, ky = (float(v) for v in solution)
        strain = eps0 + kx * y - ky * x
        case_residual = K @ solution - load
        strain_vector[case.name] = solution
        residual[case.name] = case_residual

        utilization = np.zeros(mask.shape, dtype=np.float64)
        material_reports = []
        stress_maps[case.name] = {}
        for material in materials:
            present = mask & (phi[material.name] > 0.0)
            stress = np.full(mask.shape, np.nan, dtype=np.float64)
            stress[present] = material.elastic_modulus_mpa * strain[present]
            stress_maps[case.name][material.name] = stress

            if not np.any(present):
                material_reports.append(
                    {"material": material.name, "n_pixels": 0, "present": False}
                )
                continue
            values = stress[present]
            xs, ys = x[present], y[present]
            rows, cols = np.nonzero(present)
            tension = _extreme(values, xs, ys, rows, cols, np.argmax)
            compression = _extreme(values, xs, ys, rows, cols, np.argmin)
            tensile_ratio = (
                max(tension["stress_mpa"], 0.0) / material.tensile_allowable_mpa
            )
            compressive_ratio = (
                max(-compression["stress_mpa"], 0.0)
                / material.compressive_allowable_mpa
            )
            governing = max(tensile_ratio, compressive_ratio)
            utilization[present] = np.maximum(
                utilization[present],
                np.where(
                    stress[present] >= 0.0,
                    stress[present] / material.tensile_allowable_mpa,
                    -stress[present] / material.compressive_allowable_mpa,
                ),
            )
            material_reports.append(
                {
                    "material": material.name,
                    "n_pixels": int(present.sum()),
                    "present": True,
                    "max_tension": tension,
                    "max_compression": compression,
                    "tensile_ratio": float(tensile_ratio),
                    "compressive_ratio": float(compressive_ratio),
                    "governing_ratio": float(governing),
                    "passed": bool(governing <= 1.0),
                }
            )
        utilization_maps[case.name] = utilization
        case_reports.append(
            {
                "name": case.name,
                "loads": {
                    "axial_force_n": case.axial_force_n,
                    "mx_nmm": case.mx_nmm,
                    "my_nmm": case.my_nmm,
                },
                "strain": {"eps0": eps0, "kappa_x_per_mm": kx, "kappa_y_per_mm": ky},
                "equilibrium_residual": {
                    "d_n": float(case_residual[0]),
                    "d_mx_nmm": float(case_residual[1]),
                    "d_my_nmm": float(case_residual[2]),
                },
                "materials": material_reports,
                "passed": all(m.get("passed", True) for m in material_reports),
            }
        )

    return SectionCheckResult(
        stiffness_matrix=K,
        strain_vector=strain_vector,
        residual=residual,
        stress_maps=stress_maps,
        utilization_maps=utilization_maps,
        cases=case_reports,
        materials=materials,
        load_cases=load_cases,
        pixel_spacing_mm=float(pixel_spacing_mm),
        n_section_pixels=int(mask.sum()),
    )


def report_json(result: SectionCheckResult) -> dict:
    return {
        "stiffness_matrix": {
            "units": "maps [eps0, kx(1/mm), ky(1/mm)] to [N, N*mm, N*mm]",
            "matrix": result.stiffness_matrix.tolist(),
        },
        "section": {
            "n_pixels": result.n_section_pixels,
            "pixel_spacing_mm": result.pixel_spacing_mm,
        },
        "materials": [
            {
                "name": m.name,
                "reference_density_mg_per_mm3": m.reference_density_mg_per_mm3,
                "elastic_modulus_mpa": m.elastic_modulus_mpa,
                "tensile_allowable_mpa": m.tensile_allowable_mpa,
                "compressive_allowable_mpa": m.compressive_allowable_mpa,
            }
            for m in result.materials
        ],
        "cases": result.cases,
        "passed": all(case["passed"] for case in result.cases),
        "assumptions": [
            "linear elasticity",
            "perfect bond between materials",
            "plane sections remain plane",
            "volume fractions from density/reference density; "
            "sums > 1 normalized, sums < 1 keep void",
        ],
    }
