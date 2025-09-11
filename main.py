# main.py
"""
Main entry point for J/psi and psi(2S) cross section scan study.
"""

import argparse
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d

from tqdm import tqdm

import simulation as simu


def generate_pseudo_data(
    rng,
    e_scan: np.ndarray,
    n_samples: int = simu.N_MC,
    isr: bool = simu.ISR_ON,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pseudo-data by MC sampling cross sections at scan energies.
    """
    sigma_vals = []
    sigma_errs = []
    for e in tqdm(e_scan, desc="Running MC simulation"):
        sigma = simu.mc_sigma(e, rng=rng, n_samples=n_samples, isr=isr)

        # Poisson fluctuations
        n_expected = sigma * simu.L_INT
        n_observed = rng.poisson(n_expected)
        sigma_observed = n_observed / simu.L_INT
        sigma_err = np.sqrt(n_observed) / simu.L_INT if n_observed > 0 else 1.0 / simu.L_INT

        sigma_vals.append(sigma_observed)
        sigma_errs.append(sigma_err)
    return np.array(sigma_vals), np.array(sigma_errs)


def parse_args():
    """Parse command line arguments to override defaults."""
    parser = argparse.ArgumentParser(description="J/psi and ψ(2S) cross-section scan study")

    parser.add_argument("--isr", action="store_true",
                        help="Enable ISR effects in MC (default: on)")
    parser.add_argument("--no-isr", action="store_true",
                        help="Disable ISR effects in MC (default: off)")
    parser.add_argument("--n-mc", type=int, default=simu.N_MC,
                        help=f"Number of MC samples per energy (default: {simu.N_MC})")
    parser.add_argument("--points", type=int, default=simu.N_ESCAN_POINTS,
                        help=f"Number of theory scan points (default: {simu.N_ESCAN_POINTS})")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Random seed (default: 12345)")

    return parser.parse_args()


def main():
    """Main function to run the simulation and plotting."""

    args = parse_args()

    # Resolve ISR setting
    if args.isr and args.no_isr:
        raise ValueError("Cannot set both --isr and --no-isr")
    isr_on = args.isr or (not args.no_isr and simu.ISR_ON)

    # RNG
    rng = np.random.default_rng(args.seed if args.seed is not None else simu.global_rng)

    # --- pseudo-data ---
    sigma_meas, sigma_err = generate_pseudo_data(rng,
                                                 simu.mc_energies,
                                                 n_samples=args.n_mc,
                                                 isr=isr_on)
    e_err = np.ones_like(sigma_err) * simu.energy_resolution / np.sqrt(args.n_mc)


    filename = "./simulation/theory_curves.csv"
    if not os.path.exists(filename):
    # Compute a very fine theory scan once, from which we will interpolate
        e_scan_fine = np.linspace(simu.E_MIN, simu.E_MAX, int(1e6))
        sigma_noisr_curve = simu.theory_no_isr(e_scan_fine)
        sigma_isr_curve = simu.theory_isr(e_scan_fine)

        df = pd.DataFrame({"e_scan_fine": e_scan_fine, "sigma_noisr": sigma_noisr_curve, "sigma_isr": sigma_isr_curve})
        df.to_csv(filename, index=False)
    else:
        pass

    # Load from CSV
    df = pd.read_csv(filename)
    e_scan_fine = df["e_scan_fine"].to_numpy()
    sigma_noisr_curve = df["sigma_noisr"].to_numpy()
    sigma_isr_curve = df["sigma_isr"].to_numpy()

    spline_noisr = interp1d(e_scan_fine, sigma_noisr_curve, kind="cubic")
    spline_isr = interp1d(e_scan_fine, sigma_isr_curve, kind="cubic")

    e_scan = np.linspace(simu.E_MIN, simu.E_MAX, simu.N_ESCAN_POINTS)

    # --- plotting ---
    simu.plot_scan(
        e_meas=simu.mc_energies,
        sigma_meas=sigma_meas,
        sigma_err=sigma_err,
        e_err= e_err,
        e_theory=e_scan,
        sigma_noisr=spline_noisr(e_scan),
        sigma_isr=spline_isr(e_scan),
        residuals=True,
        isr_on=isr_on,
        savepath="./figures/resonance_isr.png" if isr_on else "resonance.png",
    )

    simu.subplot_scan(
        e_meas=simu.mc_energies,
        sigma_meas=sigma_meas,
        sigma_err=sigma_err,
        e_err= e_err,
        e_theory=e_scan,
        sigma_noisr=spline_noisr(e_scan),
        sigma_isr=spline_isr(e_scan),
        resonance_windows=[(3.07, 3.13), (3.66, 3.72)],
        residuals=True,
        isr_on=isr_on,
        savepath="./figures/resonance_subplots.png"
    )


if __name__ == "__main__":
    main()
