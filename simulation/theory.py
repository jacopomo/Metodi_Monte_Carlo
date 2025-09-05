# jpsi/simulation/theory.py

"""
Theory cross section calculations (with/without ISR, Gaussian smearing).
"""

import numpy as np
from scipy.fft import fft, ifft
from tqdm import tqdm


from .constants import (
    n_quad, energy_resolution, acceptance,
    m_jpsi, gamma_jpsi, gamma_ee_jpsi,
    m_psi2s, gamma_psi2s, gamma_ee_psi2s)
from .resonance import breit_wigner_sigma
from .bhabha import build_bhabha_interpolator
from .isr import isr_radiator



def smear_gaussian_fft(sigma: np.ndarray, e_vals: np.ndarray, sigma_gauss: float) -> np.ndarray:
    """
    Smear cross section array with a Gaussian resolution using FFT convolution.

    Parameters
    ----------
    sigma : ndarray
        Cross section values on uniform energy grid.
    e_vals : ndarray
        Energy grid (must be equally spaced).
    sigma_gauss : float
        Gaussian width (standard deviation) in GeV.

    Returns
    -------
    smeared : ndarray
        Gaussian-smeared cross section on same grid.
    """
    dE = e_vals[1] - e_vals[0]
    n = len(sigma)

    # Gaussian kernel on same grid
    grid = np.arange(-n//2, n//2) * dE
    kernel = np.exp(-0.5 * (grid / sigma_gauss) ** 2)
    kernel /= np.sum(kernel)  # normalize

    # FFT convolution
    smeared = np.real(ifft(fft(sigma) * fft(kernel)))
    return smeared


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
        sigma_jpsi = acceptance() * breit_wigner_sigma(e, m_jpsi, gamma_jpsi, gamma_ee_jpsi)
        sigma_psip = acceptance * breit_wigner_sigma(e, m_psi2s, gamma_psi2s, gamma_ee_psi2s)
        sigma_res[i] = sigma_jpsi + sigma_psip + build_bhabha_interpolator(e)

    # Gaussian smearing via FFT
    return smear_gaussian_fft(sigma_res, e_vals, sigma_gauss)


def theory_isr(
    e_vals: np.ndarray,
    sigma_gauss: float = energy_resolution,
    n_quad: int = n_quad,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute theory cross section with ISR (via radiator convolution) and Gaussian smearing.

    Parameters
    ----------
    e_vals : ndarray
        Energy scan values.
    sigma_gauss : float
        Gaussian smearing width.
    n_quad : int
        Number of Gauss–Legendre quadrature nodes for ISR convolution.
    show_progress : bool
        If True, show tqdm progress bar.

    Returns
    -------
    sigma_isr : ndarray
        ISR+Gaussian smeared cross section values.
    """
    # Gauss–Legendre quadrature nodes on [0,1]
    x_nodes, w_nodes = np.polynomial.legendre.leggauss(n_quad)
    x_nodes = 0.5 * (x_nodes + 1.0)
    w_nodes = 0.5 * w_nodes

    out = np.zeros_like(e_vals)
    it = enumerate(e_vals)
    if show_progress:
        it = tqdm(it, total=len(e_vals), desc="Theory with ISR", unit="pt")

    for i, e in it:
        s_nom = e**2
        R_vals = isr_radiator(x_nodes, s_nom)
        R_norm = np.sum(w_nodes * R_vals)

        sigma_accum = 0.0
        for x, w, R in zip(x_nodes, w_nodes, R_vals):
            s_eff = (1.0 - x) * s_nom
            if s_eff <= 0:
                continue
            e_eff = np.sqrt(s_eff)

            sj = acceptance() * breit_wigner_sigma(e_eff, m_jpsi, gamma_jpsi, gamma_ee_jpsi)
            sp = acceptance() * breit_wigner_sigma(e_eff, m_psi2s, gamma_psi2s, gamma_ee_psi2s)
            sigma_accum += w * R * (sj + sp)

        if R_norm > 0:
            sigma_accum /= R_norm

        out[i] = sigma_accum + build_bhabha_interpolator(e)

    # Gaussian smearing
    return smear_gaussian_fft(out, e_vals, sigma_gauss)