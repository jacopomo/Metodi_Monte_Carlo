# main.py
"""
Main entry point for J/psi and psi(2S) cross section scan study.
"""

import numpy as np
from tqdm import tqdm

import simulation as simu


def generate_pseudo_data(
    rng,
    e_scan: np.ndarray,
    n_samples: int = simu.n_mc,
    isr: bool = simu.isr_on,
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
    isr : bool
        Whether to include ISR effects in MC simulation.

    Returns
    -------
    sigma_meas : array
        Simulated measured cross sections [nb].
    sigma_err : array
        Associated uncertainties [nb].
    """
    sigma_vals = []
    sigma_errs = []
    for e in tqdm(e_scan, desc="Running MC simulation"):
        sigma = simu.mc_sigma(e, rng=rng, n_samples=n_samples, isr=isr)

        # Poisson fluctuations
        n_expected = sigma * simu.l_int
        n_observed = rng.poisson(n_expected)
        sigma_observed = n_observed / simu.l_int
        sigma_err = np.sqrt(n_observed) / simu.l_int if n_observed > 0 else 1.0 / simu.l_int
     
        sigma_vals.append(sigma_observed)
        sigma_errs.append(sigma_err)
    return np.array(sigma_vals), np.array(sigma_errs)


def main():
    # --- pseudo-data ---
    sigma_meas, sigma_err = generate_pseudo_data(simu.global_rng, simu.mc_energies, isr=simu.isr_on)
    e_err = np.ones_like(sigma_err) * simu.energy_resolution / np.sqrt(simu.n_mc)  

    # --- theory curves on finer grid ---
    e_scan = np.linspace(simu.e_min, simu.e_max, simu.n_escan_points)
    sigma_noisr_curve = simu.theory_no_isr(e_scan)
    sigma_isr_curve = simu.theory_isr(e_scan)

    # --- plotting ---
    simu.plot_scan(
        e_meas=simu.mc_energies,
        sigma_meas=sigma_meas,
        sigma_err=sigma_err,
        e_err= e_err,
        e_theory=e_scan,
        sigma_noisr=sigma_noisr_curve,
        sigma_isr=sigma_isr_curve,
        residuals=True,
        savepath="resonance_isr.png" if simu.isr_on else "resonance.png",
    )


if __name__ == "__main__":
    main()
