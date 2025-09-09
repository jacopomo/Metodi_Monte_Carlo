# jpsi/simulation/__init__.py

"""
jpsi: Monte Carlo simulation and theory modeling for J/ψ and ψ(2S) resonances.
"""

# Expose key functions and constants at package level for convenience
from .constants import (
    M_JPSI, GAMMA_JPSI, GAMMA_EE_JPSI,
    M_PSI2S, GAMMA_PSI2S, GAMMA_EE_PSI2S,
    COS_CUT, energy_resolution, acceptance, L_INT,
    N_MC, E_MIN, E_MAX, N_ESCAN_POINTS, x_grid,
    mc_energies, global_rng, GEV2_TO_NB, ISR_ON
)

from .resonance import breit_wigner_sigma
from .background import bhabha_total
from .isr import sample_isr_x, isr_pdf
from .smearing import smear_gaussian_fft
from .montecarlo import mc_sigma
from .theory import theory_isr, theory_no_isr
from .plotting import plot_scan, subplot_scan

__all__ = [
    # Constants
    "M_JPSI", "GAMMA_JPSI", "GAMMA_EE_JPSI",
    "M_PSI2S", "GAMMA_PSI2S", "GAMMA_EE_PSI2S",
    "COS_CUT", "energy_resolution", "acceptance", "L_INT",
    "N_MC", "E_MIN", "E_MAX", "N_ESCAN_POINTS", "x_grid",
    "mc_energies", "global_rng", "GEV2_TO_NB", "ISR_ON",

    # Resonances
    "breit_wigner_sigma",

    # Background
    "bhabha_total",

    # ISR
    "sample_isr_x", "isr_pdf",

    # Smearing
    "smear_gaussian_fft",

    # Monte Carlo
    "mc_sigma",

    # Theory
    "theory_isr", "theory_no_isr",

    # Plotting
    "plot_scan", "subplot_scan",
]
