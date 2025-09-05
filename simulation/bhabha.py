# jpsi/simulation/bhabha.py

"""
Bhabha scattering cross section and precomputed interpolator.
"""

import numpy as np
from scipy import interpolate

from .constants import alpha, gev2_to_nb, cos_cut, e_min, e_max, n_escan_points, acceptance


def bhabha_integrand(cos_theta: float, s: float) -> float:
    """
    Differential Bhabha scattering integrand.

    Parameters
    ----------
    cos_theta : float
        cos(theta) value.
    s : float
        Mandelstam s = E_cm^2 [GeV^2].

    Returns
    -------
    float
        Integrand value.
    """
    t = -0.5 * s * (1 - cos_theta)
    u = -0.5 * s * (1 + cos_theta)
    return (s**2 + u**2) / (t**2) + (t**2 + u**2) / (s**2) + 2 * u**2 / (s * t)


def bhabha_total(E_cm: float, cos_max: float = cos_cut) -> float:
    """
    Total Bhabha cross section (nb) with |cos(theta)| < cos_min cut.

    Parameters
    ----------
    E_cm : float
        Center-of-mass energy [GeV].
    cos_min : float
        Angular cut (default = cos_cut from constants).

    Returns
    -------
    sigma : float
        Cross section [nb].
    """
    s = E_cm**2
    integral = acceptance(dist_func=lambda c: bhabha_integrand(c, s), cos_max=cos_max, norm=False)
    return (np.pi * alpha**2 / s) * integral * gev2_to_nb