# jpsi/simulation/background.py

"""
Bhabha scattering cross section and precomputed interpolator.
"""

import numpy as np

from .constants import ALPHA, GEV2_TO_NB, COS_CUT, acceptance


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


def bhabha_total(e_cm: float, cos_max: float = COS_CUT) -> float:
    """
    Total Bhabha cross section (nb) with |cos(theta)| < cos_min cut.

    Parameters
    ----------
    E_cm : float
        Center-of-mass energy [GeV].
    cos_min : float
        Angular cut (default = COS_CUT from constants).

    Returns
    -------
    sigma : float
        Cross section [nb].
    """
    s = e_cm**2
    integral = acceptance(dist_func=lambda c: bhabha_integrand(c, s), cos_max=cos_max, norm=False)
    return (np.pi * ALPHA**2 / s) * integral * GEV2_TO_NB
