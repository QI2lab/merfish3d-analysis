"""Shared point-spread-function generation for qi2lab microscopes."""

import numpy as np

QI2LAB_DEFAULT_NA = 1.35
QI2LAB_DEFAULT_IMMERSION_RI = 1.51
QI2LAB_SAMPLE_RI = 1.47
QI2LAB_PSF_NX = 51
QI2LAB_EXCITATION_WAVELENGTHS_UM = (0.488, 0.561, 0.635)
QI2LAB_EMISSION_WAVELENGTHS_UM = (0.520, 0.580, 0.670)


def generate_qi2lab_psf(
    *,
    z_depth: int,
    voxel_size_zyx_um: tuple[float, float, float],
    emission_wavelength_um: float,
    na: float = QI2LAB_DEFAULT_NA,
    immersion_ri: float = QI2LAB_DEFAULT_IMMERSION_RI,
) -> np.ndarray:
    """Generate a normalized vectorial PSF with qi2lab microscope parameters."""
    from psfmodels import make_psf

    if z_depth < 1:
        raise ValueError(f"PSF Z depth must be positive; received {z_depth}.")
    z_size_um, y_size_um, _x_size_um = voxel_size_zyx_um
    psf = make_psf(
        z=z_depth,
        nx=QI2LAB_PSF_NX,
        dxy=y_size_um,
        dz=z_size_um,
        NA=na,
        wvl=emission_wavelength_um,
        ns=QI2LAB_SAMPLE_RI,
        ni=immersion_ri,
        ni0=immersion_ri,
        model="vectorial",
    ).astype(np.float32)
    psf_sum = float(np.sum(psf, dtype=np.float64))
    if not np.isfinite(psf_sum) or psf_sum <= 0:
        raise ValueError(f"Generated PSF has invalid sum {psf_sum!r}.")
    return (psf / psf_sum).astype(np.float32, copy=False)
