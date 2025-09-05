# jpsi/simulation/isr.py
"""
Initial-State Radiation (ISR) radiator functions and sampling.
Bonneau-Martin / Kuraev-Fadin O(alpha).
"""

import numpy as np
from functools import lru_cache
from typing import Tuple

from .constants import alpha, m_e_gev


def isr_radiator(x, s):
    """
    ISR radiator function at O(alpha).

    Parameters
    ----------
    x : array-like
        Fractional energy loss.
    s : float
        Nominal invariant mass squared [GeV^2].

    Returns
    -------
    ndarray
        Radiator values (unnormalized).
    """
    x = np.asarray(x)
    L = np.log(s / m_e_gev**2)
    pref = alpha / np.pi
    safe_x = np.where(x <= 0.0, 1e-300, x) # avoid division by zero
    val = pref * ((1 + (1 - x)**2) / safe_x * (L - 1) - x)
    return np.clip(val, 0.0, None)


@lru_cache(maxsize=256)
def build_isr_cdf_cached(key: Tuple[float, int]):
    """
    Build and cache ISR CDF on a log-spaced x-grid.

    Parameters
    ----------
    key : tuple
        (s_nom, nx)

    Returns
    -------
    x_grid : ndarray
    cdf : ndarray
    """
    s_nom, nx = key
    x_grid = np.logspace(-5.5, np.log10(0.999), nx)
    w_vals = isr_radiator(x_grid, s_nom)
    integral = np.trapz(w_vals, x_grid)
    if integral <= 0:
        pdf = np.ones_like(w_vals) / w_vals.size
    else:
        pdf = w_vals / integral
    dx = np.diff(np.concatenate(([0.0], x_grid)))
    cdf = np.cumsum(pdf * dx)
    cdf /= cdf[-1]
    return x_grid, cdf


def sample_isr_x(s_nom: float, rng, n_samples: int,
                 nx: int = 4000, round_digits: int = 6):
    """
    Sample ISR fractions x.

    Parameters
    ----------
    s_nom : float
        Nominal invariant mass squared [GeV^2].
    rng : np.random.Generator
        Random number generator.
    n_samples : int
        Number of samples.
    nx : int
        Number of x grid points for CDF.
    round_digits : int
        Precision for cache key.

    Returns
    -------
    ndarray
        ISR energy fractions x.
    """
    key = (round(float(s_nom), round_digits), nx)
    x_grid, cdf_grid = build_isr_cdf_cached(key)
    us = rng.random(n_samples)
    return np.interp(us, cdf_grid, x_grid)
