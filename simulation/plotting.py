# jpsi/plotting.py

"""
Plotting utilities for J/psi and psi' cross section studies.
"""

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


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
    residuals: bool = True,
    isr_on: bool = True,
    title: str = "J/ψ and ψ(2S) Cross Section Scan",

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
    residuals : bool
        Whether to plot residuals below main plot.
    isr_on : bool
        Whether ISR effects are included in the MC pseudo-data.
    title : str
        Title of the plot.

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

def subplot_scan(
    e_meas: np.ndarray,
    sigma_meas: np.ndarray,
    sigma_err: np.ndarray,
    e_err: np.ndarray,
    e_theory: np.ndarray,
    sigma_noisr: np.ndarray,
    sigma_isr: np.ndarray,
    resonance_windows: list[tuple[float, float]],
    savepath: Optional[str] = None,
    show: bool = True,
    residuals: bool = True,
    isr_on: bool = True,
    title: str = "Zoomed-in Resonance Scans",
) -> plt.Figure:
    """
    Plot two zoomed-in subplots around resonances with residuals.

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
    resonance_windows : list of (float, float)
        List of (emin, emax) ranges for zooming around resonances.
        Should contain two entries, e.g. [(3.05, 3.14), (3.65, 3.72)].
    sigma_fit : array, optional
        Optional fitted theory curve to overlay.
    savepath : str, optional
        If provided, save figure to this path.
    show : bool, default=True
        Whether to display the plot interactively.
    residuals : bool
        Whether to plot residuals below main plot.
    isr_on : bool
        Whether ISR effects are included in the MC pseudo-data.
    title : str
        Title of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    """

    n_res = len(resonance_windows)
    if n_res != 2:
        raise ValueError("resonance_windows must contain exactly two (emin, emax) tuples")

    fig, axes = plt.subplots(
        n_res, 2 if residuals else 1,
        figsize=(12, 5 * n_res),
        gridspec_kw={"width_ratios": [3, 1]} if residuals else None,
        sharey=False
    )

    if n_res == 1:
        axes = [axes]  # normalize structure

    for i, (emin, emax) in enumerate(resonance_windows):
        if residuals:
            ax_main, ax_res = axes[i]
        else:
            ax_main = axes[i]
            ax_res = None

        # restrict data to window
        mask_meas = (e_meas >= emin) & (e_meas <= emax)
        mask_theo = (e_theory >= emin) & (e_theory <= emax)

        e_m = e_meas[mask_meas]
        sig_m = sigma_meas[mask_meas]
        sig_err = sigma_err[mask_meas]
        e_err_m = e_err[mask_meas]
        e_t = e_theory[mask_theo]
        sig_no = sigma_noisr[mask_theo]
        sig_isr = sigma_isr[mask_theo]

        # MC data
        ax_main.errorbar(
            e_m, sig_m,
            yerr=sig_err, xerr=e_err_m,
            fmt=".", color="red", label="MC pseudo-data", capsize=2
        )

        # theory curves
        ax_main.plot(e_t, sig_no, "--", color="black", label="Theory (no ISR)")
        ax_main.plot(e_t, sig_isr, "-", color="blue", label="Theory (with ISR)")

        ax_main.set_ylabel("Cross section (nb)")
        ax_main.set_xlim(emin, emax)
        ax_main.set_yscale("log")
        ax_main.grid(True, ls="--", alpha=0.6)
        ax_main.legend()
        ax_main.set_title(f"{title}: {emin:.3f}–{emax:.3f} GeV")

        if residuals and ax_res is not None and len(e_m) > 0:
            if isr_on:
                sigma_interp = np.interp(e_m, e_t, sig_isr)
            else:
                sigma_interp = np.interp(e_m, e_t, sig_no)

            res = (sig_m - sigma_interp) / sig_err
            ax_res.axhline(0, color="black", lw=1)
            ax_res.errorbar(
                e_m, res, yerr=np.ones_like(sig_err),
                fmt="o", color="purple", capsize=2
            )
            ax_res.set_xlabel("CM Energy (GeV)")
            ax_res.set_ylabel("Normalized residuals")
            ax_res.grid(True, ls="--", alpha=0.6)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    if show:
        plt.show()

    return fig
