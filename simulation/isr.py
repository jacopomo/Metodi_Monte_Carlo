# jpsi/simulation/isr.py

"""
Initial-State Radiation (ISR) radiator distribution at O(alpha).
Bonneau-Martin / Kuraev-Fadin formulation.

Here we construct a universal normalized PDF and CDF that are independent
of the nominal center-of-mass energy s (lowest order).
"""

from typing import Tuple
import numpy as np

from .constants import x_grid

def isr_pdf(x: np.ndarray) -> np.ndarray:
    """
    Normalized ISR probability density function (universal).

    The dependence on s cancels out, so this PDF is the same
    for all energies.

    Parameters
    ----------
    x : array-like
        Fractional energy loss (0 < x < 1).

    Returns
    -------
    ndarray
        Normalized PDF values.
    """
    x = np.asarray(x)
    safe_x = np.where(x <= 0.0, 1e-300, x)
    base = (1 + (1 - x) ** 2) / safe_x
    # normalize numerically on [xmin, xmax]
    grid = x_grid
    base_grid = (1 + (1 - grid) ** 2) / grid
    integral = np.trapezoid(base_grid, grid)
    return base / integral


def isr_cdf() -> Tuple[np.ndarray, np.ndarray]:
    """
    Build ISR CDF on a log-spaced x-grid.

    Returns
    -------
    x_grid : ndarray
        Energy-loss fractions.
    cdf : ndarray
        Normalized cumulative distribution.
    """
    pdf_vals = isr_pdf(x_grid)
    dx = np.diff(np.concatenate(([0.0], x_grid)))
    cdf = np.cumsum(pdf_vals * dx)
    cdf /= cdf[-1]
    return x_grid, cdf


def sample_isr_x(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """
    Sample ISR energy-loss fractions x ~ PDF.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n_samples : int
        Number of samples.

    Returns
    -------
    ndarray
        ISR energy-loss fractions x.
    """
    _, cdf = isr_cdf()
    us = rng.random(size=n_samples)
    return np.interp(us, cdf, x_grid)
