# jpsi/simulation/montecarlo.py

"""
Monte Carlo cross section estimators with ISR and Gaussian smearing.
"""

import numpy as np

from .constants import (
    n_mc, energy_resolution, global_rng, acceptance,
    m_jpsi, gamma_jpsi, gamma_ee_jpsi,
    m_psi2s, gamma_psi2s, gamma_ee_psi2s)
from .resonance import breit_wigner_sigma
from .bhabha import bhabha_total
from .isr import sample_isr_x

def mc_sigma(
    e_nom: float,
    rng=global_rng,
    n_samples: int = n_mc,
    isr: bool = True,
) -> float:
    """
    Monte Carlo estimate of cross section at nominal energy with ISR and Gaussian smearing.
    
    Parameters
    ----------
    e_nom : float
        Nominal center-of-mass energy [GeV].
    rng : np.random.Generator
        Random number generator.
    n_samples : int
        Number of MC samples.
    isr : bool
        If True, include ISR effects.
    
    Returns
    -------
    float
        Estimated cross section [nb] at e_nom including ISR and smearing.
    """
    if isr:
        # sample ISR fractions
        x_samp = sample_isr_x(rng, n_samples)

        # effective energies
        e_eff = np.sqrt((1.0 - x_samp) * e_nom**2)

    else:
        e_eff = np.full(n_samples, e_nom)

    # Gaussian smear
    e_smeared = rng.normal(loc=e_eff, scale=energy_resolution)

    # Resonances
    sigma_jpsi = acceptance() * breit_wigner_sigma(e_smeared, m_jpsi, gamma_jpsi, gamma_ee_jpsi)
    sigma_psip = acceptance() * breit_wigner_sigma(e_smeared, m_psi2s, gamma_psi2s, gamma_ee_psi2s)
    return float(np.mean(sigma_jpsi + sigma_psip) + bhabha_total(e_nom))