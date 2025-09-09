# jpsi/simulation/resonance.py

"""
Resonance cross section model (Breit-Wigner).
"""

import numpy as np

from .constants import GEV2_TO_NB

def breit_wigner_sigma(e: np.ndarray, m: float, gamma: float, gamma_ee: float) -> np.ndarray:
    """
    Breit-Wigner cross section for a vector resonance.

    Parameters
    ----------
    e : array-like
        Center-of-mass energy [GeV].
    m : float
        Resonance mass [GeV].
    gamma : float
        Resonance width [GeV].
    gamma_ee : float
        Electron partial width [GeV].

    Returns
    -------
    sigma : ndarray
        Cross section in nb.
    """
    # PDG Breit–Wigner formula (simplified form, no interference)
    s = e**2
    num = 12 * np.pi * gamma_ee**2
    den = (s - m**2) ** 2 + (m * gamma) ** 2
    return (num / den) * GEV2_TO_NB
