# jpsi/plotting.py

"""
Plotting utilities for J/psi and psi' cross section studies.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from .constants import isr_on

def plot_scan(
    e_meas: np.ndarray,
    sigma_meas: np.ndarray,
    sigma_err: np.ndarray,
    e_err: np.ndarray,
    e_theory: np.ndarray,
    sigma_noisr: np.ndarray,
    sigma_isr: np.ndarray,
    savepath: Optional[str] = None,
    show: bool = True,
    logy: bool = True,
    title: str = "J/ψ and ψ(2S) Cross Section Scan",
    residuals: bool = True,
) -> plt.Figure:
    """
    Plot MC pseudo-data together with theory curves.

    Parameters
    ----------
    e_meas : array
        Measured (pseudo-data) energies [GeV].
    sigma_meas : array
        Measured cross sections [nb].
    sigma_err : array
        Uncertainties on measured cross sections [nb].
    e_err : array
        Uncertainties on measured energies [GeV].
    e_theory : array
        Theory energy grid [GeV].
    sigma_noisr : array
        Theory curve without ISR.
    sigma_isr : array
        Theory curve with ISR.
    sigma_fit : array, optional
        Optional fitted theory curve to overlay.
    savepath : str, optional
        If provided, save figure to this path.
    show : bool, default=True
        Whether to display the plot interactively.
    logy : bool, default=True
        Use logarithmic y-axis scale.
    title : str
        Title of the plot.
    residuals : bool
        Whether to plot residuals below main plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    if residuals:
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax_main = plt.subplots(figsize=(10, 6))
        ax_res = None

    # MC pseudo-data
    ax_main.errorbar(
        e_meas,
        sigma_meas,
        yerr=sigma_err,
        xerr=e_err,  
        fmt=".",
        color="red",
        label="MC pseudo-data",
        capsize=2,
    )

    # Theory curves
    ax_main.plot(e_theory, sigma_noisr, "--", color="black", label="Theory (no ISR)")
    ax_main.plot(e_theory, sigma_isr, "-", color="blue", label="Theory (with ISR)")

    ax_main.set_ylabel("Cross section (nb)")
    ax_main.set_title(title)
    if logy:
        ax_main.set_yscale("log")
    ax_main.grid(True, which="both", ls="--", alpha=0.6)
    ax_main.legend()

    if residuals and ax_res is not None:
        if isr_on:
            sigma_interp = np.interp(e_meas, e_theory, sigma_isr)
        else:
            sigma_interp = np.interp(e_meas, e_theory, sigma_noisr)

        res = (sigma_meas - sigma_interp) / sigma_err

        ax_res.axhline(0, color="black", lw=1)
        ax_res.errorbar(
            e_meas,
            res,
            yerr=np.ones_like(sigma_err),
            fmt="o",
            color="purple",
            capsize=2,
        )
        ax_res.set_xlabel("CM Energy (GeV)")
        ax_res.set_ylabel("Normalized residuals (nb)")
        ax_res.grid(True, ls="--", alpha=0.6)

        print(f"Chi2: {np.sum(res**2):.2f}")
        print(f"Degrees of freedom: {len(res) - 1}")
        print(f"Chi2/ndf: {np.sum(res**2)/(len(res)-1):.2f}")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    if show:
        plt.show()

    return fig