# jpsi/simulation/__init__.py

"""
jpsi: Monte Carlo simulation and theory modeling for J/ψ and ψ(2S) resonances.
"""

# Expose key functions and constants at package level for convenience
from .constants import (
    m_jpsi, gamma_jpsi, gamma_ee_jpsi,
    m_psi2s, gamma_psi2s, gamma_ee_psi2s, 
    cos_cut, energy_resolution, acceptance, l_int,
    n_mc, e_min, e_max, n_escan_points, x_grid,
    mc_energies, global_rng, gev2_to_nb, isr_on
)

from .resonance import breit_wigner_sigma
from .bhabha import bhabha_total
from .isr import sample_isr_x, isr_pdf
from .smearing import smear_gaussian_fft
from .montecarlo import mc_sigma
from .theory import theory_isr, theory_no_isr
from .plotting import plot_scan

__all__ = [
    # Constants
    "m_jpsi", "gamma_jpsi", "gamma_ee_jpsi",
    "m_psi2s", "gamma_psi2s", "gamma_ee_psi2s",
    "cos_cut", "energy_resolution", "acceptance", "l_int",
    "n_mc", "e_min", "e_max", "n_escan_points", "x_grid",
    "mc_energies", "global_rng", "gev2_to_nb", "isr_on" 

    # Resonances
    "breit_wigner_sigma",

    # Bhabha
    "bhabha_total",

    # ISR
    "sample_isr_x", "isr_pdf",

    # Smearing
    "smear_gaussian_fft",

    # Monte Carlo
    "mc_sigma"

    # Theory
    "theory_isr", "theory_no_isr",

    # Plotting
    "plot_scan",
]
