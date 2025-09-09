# jpsi/simulation/theory.py

"""
Theory cross section calculations (with/without ISR, Gaussian smearing).
"""

import numpy as np
from tqdm import tqdm


from .constants import (
    energy_resolution, acceptance,
    M_JPSI, GAMMA_JPSI, GAMMA_EE_JPSI, x_grid,
    M_PSI2S, GAMMA_PSI2S, GAMMA_EE_PSI2S)
from .resonance import breit_wigner_sigma
from .background import bhabha_total
from .isr import isr_pdf
from .smearing import smear_gaussian_fft

def theory_no_isr(
    e_vals: np.ndarray,
    sigma_gauss: float = energy_resolution,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute theory cross section with Gaussian smearing but no ISR.

    Parameters
    ----------
    e_vals : ndarray
        Energy scan values.
    sigma_gauss : float
        Gaussian smearing width.
    show_progress : bool
        If True, show tqdm progress bar.

    Returns
    -------
    sigma_no_isr : ndarray
        Smeared cross section values at e_vals.
    """
    # Evaluate unsmeared resonance cross sections
    sigma_res = np.zeros_like(e_vals)
    it = enumerate(e_vals)
    if show_progress:
        it = tqdm(it, total=len(e_vals), desc="Theory no ISR", unit="pt")

    for i, e in it:
        sigma_jpsi = acceptance() * breit_wigner_sigma(e, M_JPSI, GAMMA_JPSI, GAMMA_EE_JPSI)
        sigma_psip = acceptance() * breit_wigner_sigma(e, M_PSI2S, GAMMA_PSI2S, GAMMA_EE_PSI2S)
        sigma_res[i] = sigma_jpsi + sigma_psip + bhabha_total(e)

    # Gaussian smearing via FFT
    return smear_gaussian_fft(sigma_res, e_vals, sigma_gauss)

def theory_isr(
    e_vals: np.ndarray,
    sigma_gauss: float = energy_resolution,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute theory cross section with ISR (via precomputed ISR PDF) and Gaussian smearing.

    Parameters
    ----------
    e_vals : ndarray
        Energy scan values.
    sigma_gauss : float
        Gaussian smearing width.
    show_progress : bool
        If True, show tqdm progress bar.

    Returns
    -------
    sigma_isr : ndarray
        ISR+Gaussian smeared cross section values.
    """

    out = np.zeros_like(e_vals)
    it = enumerate(e_vals)
    if show_progress:
        it = tqdm(it, total=len(e_vals), desc="Theory with ISR", unit="pt")

    for i, e_nom in it:
        e_eff = np.sqrt((1.0 - x_grid) * e_nom**2)

        # Evaluate resonances
        sigma_jpsi = breit_wigner_sigma(e_eff, M_JPSI, GAMMA_JPSI, GAMMA_EE_JPSI)
        sigma_psip = breit_wigner_sigma(e_eff, M_PSI2S, GAMMA_PSI2S, GAMMA_EE_PSI2S)

        # Convolve with ISR PDF using trapezoidal integration
        sigma_isr_val = np.trapz(isr_pdf(x_grid) * (sigma_jpsi + sigma_psip), x_grid)

        # Add Bhabha contribution
        out[i] =  acceptance() * sigma_isr_val + bhabha_total(e_nom)

    # Gaussian smearing
    return smear_gaussian_fft(out, e_vals, sigma_gauss)
