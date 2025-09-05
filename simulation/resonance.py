# jpsi/simulation/resonance.py

"""
Resonance cross section model (Breit-Wigner).
"""

import numpy as np
from scipy import integrate

from .constants import gev2_to_nb, cos_cut

def breit_wigner_sigma(E: np.ndarray, M: float, Gamma: float, Gamma_ee: float) -> np.ndarray:
    """
    Breit-Wigner cross section for a vector resonance.

    Parameters
    ----------
    E : array-like
        Center-of-mass energy [GeV].
    M : float
        Resonance mass [GeV].
    Gamma : float
        Resonance width [GeV].
    Gamma_ee : float
        Electron partial width [GeV].

    Returns
    -------
    sigma : ndarray
        Cross section in nb.
    """
    # PDG Breit–Wigner formula (simplified form, no interference)
    s = E**2
    num = 12 * np.pi * Gamma_ee**2
    den = (s - M**2) ** 2 + (M * Gamma) ** 2
    return (num / den) * gev2_to_nb  