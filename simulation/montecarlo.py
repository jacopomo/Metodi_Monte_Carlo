# jpsi/simulation/montecarlo.py

"""
Monte Carlo cross section estimators with ISR and Gaussian smearing.
"""

import numpy as np

from .constants import (
    n_mc, n_quad, energy_resolution, global_rng, acceptance,
    m_jpsi, gamma_jpsi, gamma_ee_jpsi,
    m_psi2s, gamma_psi2s, gamma_ee_psi2s)
from .resonance import breit_wigner_sigma
from .bhabha import bhabha_total
from .isr import sample_isr_x

def mc_sigma_with_isr(
    e_nom: float,
    rng=global_rng,
    n_samples: int = n_mc,
) -> float:
    """
    Stochastic Monte Carlo expected cross section at nominal energy e_nom:
    - sample ISR fractions x
    - compute ISR-reduced energies E_eff_isr = sqrt((1-x) * e_nom^2)
    - sample Gaussian-smeared energies around E_eff_isr
    - evaluate BW resonances at smeared energies
    - average and add unsmeared Bhabha(E_nom)
    """

    # sample ISR fractions
    x_samp = sample_isr_x(e_nom**2, rng, n_samples)

    # effective energies
    e_eff_isr = np.sqrt((1.0 - x_samp) * e_nom**2)

    # Gaussian smear
    e_smeared = rng.normal(loc=e_eff_isr, scale=energy_resolution)

    # Resonances
    sigma_jpsi = acceptance() * breit_wigner_sigma(e_smeared, m_jpsi, gamma_jpsi, gamma_ee_jpsi)
    sigma_psip = acceptance() * breit_wigner_sigma(e_smeared, m_psi2s, gamma_psi2s, gamma_ee_psi2s)
    return float(np.mean(sigma_jpsi + sigma_psip) + bhabha_total(e_nom))