# main.py
"""
Main entry point for J/psi and psi(2S) cross section scan study.
"""

import numpy as np

import simulation as simu


def generate_pseudo_data(
    rng,
    e_scan: np.ndarray,
    n_samples: int = simu.n_mc,
    frac_err: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pseudo-data by MC sampling cross sections at scan energies.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    e_scan : array
        Energies [GeV] at which to simulate measurements.
    n_samples : int
        Number of MC samples per energy.
    frac_err : float
        Fractional uncertainty (controls pseudo error bars).

    Returns
    -------
    sigma_meas : array
        Simulated measured cross sections [nb].
    sigma_err : array
        Associated uncertainties [nb].
    """
    sigma_vals = []
    sigma_errs = []
    for e in e_scan:
        sigma = simu.mc_sigma_with_isr(e, rng=rng, n_samples=n_samples)
        err = frac_err * sigma
        # smear pseudo-data around true value
        meas = rng.normal(loc=sigma, scale=err)
        sigma_vals.append(meas)
        sigma_errs.append(err)
    return np.array(sigma_vals), np.array(sigma_errs)


def main():
    # --- energy grid for scan (measured points) ---
    e_scan = np.linspace(simu.e_min, simu.e_max, simu.n_escan_points)

    # --- pseudo-data ---
    sigma_meas, sigma_err = generate_pseudo_data(simu.global_rng, e_scan)

    # --- theory curves on finer grid ---
    e_theory = np.linspace(simu.e_min, simu.e_max, 400)
    sigma_noisr_curve = simu.theory_no_isr(e_theory)
    sigma_isr_curve = simu.theory_isr(e_theory)

    # --- plotting ---
    simu.plot_scan(
        e_meas=e_scan,
        sigma_meas=sigma_meas,
        sigma_err=sigma_err,
        e_theory=e_theory,
        sigma_noisr=sigma_noisr_curve,
        sigma_isr=sigma_isr_curve,
        residuals=True,
        savepath="scan_plot.png",
    )


if __name__ == "__main__":
    main()
