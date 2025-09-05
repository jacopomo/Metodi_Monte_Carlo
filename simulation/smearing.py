# jpsi/simulation/smearing.py
"""
Smearing utilities (Gaussian convolution).
"""

import numpy as np
from scipy import signal


def smear_gaussian_fft(y_vals: np.ndarray, e_vals: np.ndarray,
                       sigma_e: float) -> np.ndarray:
    """
    Gaussian smearing by FFT-based convolution.

    Parameters
    ----------
    y_vals : ndarray
        Function values to smear.
    e_vals : ndarray
        Energy grid (must be uniform).
    sigma_e : float
        Gaussian sigma [GeV].

    Returns
    -------
    ndarray
        Smeared values.
    """
    if len(e_vals) < 2:
        return y_vals
    dE = e_vals[1] - e_vals[0]
    pad = int(max(1, np.ceil(3 * sigma_e / dE)))
    y_pad = np.pad(y_vals, pad, mode="edge")
    idx = np.arange(-len(y_pad) // 2, len(y_pad) // 2)
    kernel = np.exp(-0.5 * ((idx * dE) / sigma_e) ** 2)
    kernel /= kernel.sum()
    conv = signal.fftconvolve(y_pad, kernel, mode="same")
    conv = conv[pad:-pad]
    return np.clip(conv, 1e-12, None)