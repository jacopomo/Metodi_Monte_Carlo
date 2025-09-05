#!/usr/bin/env python3
"""
jpsi.py

Simulate an e+e- energy scan around the J/ψ and ψ(2S) resonances,
including Initial-State Radiation (ISR) using a Kuraev-Fadin / Bonneau-Martin
O(alpha) radiator, and Gaussian beam/detector smearing.

Features:
- Monte Carlo ISR sampling (stochastic)
- Deterministic theory convolution (ISR via quadrature, then Gaussian smear)
- Deterministic-style MC (mc_deterministic_like) to validate MC ↔ theory equality
- CLI flags for fast testing and exporting results
- Progress bars for long loops
- Convergence test utility
"""

from functools import lru_cache
import argparse
import os
import sys
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, signal, interpolate

# -----------------
# Constants & default parameters
# -----------------
ALPHA = 1 / 137.035999084
GEV2_TO_NB = 389379.338
M_E_GEV = 0.00051099895

# Resonance PDG-like values
M_JPSI = 3.0969
GAMMA_JPSI = 0.0000929
GAMMAEE_JPSI = 5.547e-6

M_PSIP = 3.6861
GAMMA_PSIP = 0.000294
GAMMAEE_PSIP = 2.33142e-6

# Acceptance & resolution
COS_CUT = 0.6
SIGMA_BEAM = 0.003
SIGMA_DETECTOR = 0.003
SIGMA_COMBINED = float(np.sqrt(SIGMA_BEAM**2 + SIGMA_DETECTOR**2))

# Luminosity/exposure (used for Poisson fluctuations)
L_INSTANT_CM2_S = 2e30
L_INSTANT_NB_INV_S = L_INSTANT_CM2_S * 1e-33
T_PER_POINT = 100 * 3600
L_INT = L_INSTANT_NB_INV_S * T_PER_POINT
EFFICIENCY = 1.0

# MC defaults (can be overridden via CLI)
DEFAULT_N_MC_SAMPLES = 100000
DEFAULT_N_QUAD = 400
DEFAULT_EGRID_POINTS = 10000

# RNG
GLOBAL_RNG = np.random.default_rng(12345)

# -----------------
# Energy scan points
# -----------------
D_ES = 0.022  # step size for fine scan regions
E_SCAN = np.concatenate([
    np.linspace(2.8, 3.0969 - D_ES, 12),                     # below J/ψ
    np.linspace(3.0969 - D_ES, 3.0969 + D_ES, 50),          # around J/ψ
    np.linspace(3.0969 + D_ES, 3.6861 - D_ES, 16),           # between resonances
    np.linspace(3.6861 - D_ES, 3.6861 + D_ES, 20),           # around ψ'
    np.linspace(3.6861 + D_ES, 3.80, 6)                      # above ψ'
])

# -----------------
# Physics helpers
# -----------------
def bhabha_integrand(cos_theta: float, s: float) -> float:
    """Integrand for Bhabha scattering differential cross section."""
    t = -0.5 * s * (1 - cos_theta)
    u = -0.5 * s * (1 + cos_theta)
    return (s**2 + u**2) / (t**2) + (t**2 + u**2) / (s**2) + 2 * u**2 / (s * t)


def bhabha_total(E_cm: float, cos_min: float = COS_CUT) -> float:
    """Integrated Bhabha cross section (nb) for CM energy E_cm [GeV]."""
    s = E_cm**2
    integral, _ = integrate.quad(
        lambda c: bhabha_integrand(c, s), -cos_min, cos_min, epsrel=1e-9
    )
    return (np.pi * ALPHA**2 / s) * integral * GEV2_TO_NB


def build_bhabha_interpolator(e_min: float = 2.6,
                              e_max: float = 3.9,
                              n_points: int = 2000):
    """Precompute a cubic interpolator for the Bhabha cross section."""
    energies = np.linspace(e_min, e_max, n_points)
    values = np.array([bhabha_total(E) for E in energies])
    return interpolate.interp1d(energies, values,
                                kind="cubic",
                                bounds_error=False,
                                fill_value="extrapolate")


def breit_wigner_sigma(E, M, Gamma, Gamma_ee):
    """Breit-Wigner cross section (nb) for vector resonance evaluated at E (GeV)."""
    E = np.asarray(E)
    s = E**2
    numer = 12 * np.pi * Gamma_ee**2
    denom = (s - M**2) ** 2 + (M**2) * (Gamma**2)
    return numer / denom * GEV2_TO_NB


def acceptance_factor(cos_max: float = COS_CUT) -> float:
    """Fractional acceptance from angular cut."""
    num = integrate.quad(lambda u: 1 + u**2, -cos_max, cos_max)[0]
    den = integrate.quad(lambda u: 1 + u**2, -1, 1)[0]
    return num / den


B_ACC = acceptance_factor()

# Build Bhabha interpolator once and make it global (cheap to compute once)
BHABHA_INTERP = build_bhabha_interpolator()


# -----------------
# ISR radiator & sampling (Bonneau-Martin / Kuraev-Fadin O(alpha))
# -----------------
def isr_radiator_oalpha(x, s):
    """
    Return O(alpha) radiator function values for array-like x and invariant s.
    This is unnormalized and will be normalized externally to create PDFs.
    """
    x = np.asarray(x)
    L = np.log(s / (M_E_GEV**2))
    pref = ALPHA / np.pi
    safe_x = np.where(x <= 0.0, 1e-300, x)
    val = pref * ((1 + (1 - x)**2) / safe_x * (L - 1) - x)
    return np.clip(val, 0.0, None)


@lru_cache(maxsize=256)
def build_isr_cdf_cached(key: Tuple[float, int]):
    """
    Build & cache ISR CDF on a log-spaced x-grid to resolve near-zero singularity.
    key = (s_nom_rounded, nx)
    returns (x_grid, cdf)
    """
    s_nom_rounded, nx = key
    s_nom = float(s_nom_rounded)
    x_grid = np.logspace(-5.5, np.log10(0.999), nx)
    w_vals = isr_radiator_oalpha(x_grid, s_nom)
    integral = np.trapezoid(w_vals, x_grid)
    if integral <= 0:
        pdf = np.ones_like(w_vals) / w_vals.size
    else:
        pdf = w_vals / integral
    dx = np.diff(np.concatenate(([0.0], x_grid)))
    cdf = np.cumsum(pdf * dx)
    cdf /= cdf[-1]
    return x_grid, cdf


def sample_isr_x(s_nom: float,
                 rng,
                 n_samples: int,
                 nx: int = 4000,
                 round_digits: int = 6):
    """Sample n_samples ISR fractions x from the cached CDF for s_nom."""
    key = (round(float(s_nom), round_digits), nx)
    x_grid, cdf_grid = build_isr_cdf_cached(key)
    us = rng.random(n_samples)
    inv = interpolate.interp1d(cdf_grid, x_grid,
                              bounds_error=False,
                              fill_value=(x_grid[0], x_grid[-1]))
    return inv(us)


# -----------------
# Smearing helpers
# -----------------
def smear_gaussian_fft(y_vals: np.ndarray, e_vals: np.ndarray,
                       sigma_e: float) -> np.ndarray:
    """
    FFT-based Gaussian convolution, normalized kernel, edge padding.
    Returns clipped array (no zeros).
    """
    if len(e_vals) < 2:
        return y_vals  # cannot smear a single point, just return it
    dE = e_vals[1] - e_vals[0]
    pad = int(max(1, np.ceil(3 * sigma_e / dE)))
    y_pad = np.pad(y_vals, pad, mode="edge")
    idx = np.arange(-len(y_pad) // 2, len(y_pad) // 2)
    kernel = np.exp(-0.5 * ((idx * dE) / sigma_e)**2)
    kernel /= kernel.sum()
    conv = signal.fftconvolve(y_pad, kernel, mode="same")
    conv = conv[pad:-pad]
    return np.clip(conv, 1e-12, None)


# -----------------
# Monte Carlo functions
# -----------------
def mc_sigma_with_isr(e_nom: float,
                      rng=None,
                      n_samples: int = DEFAULT_N_MC_SAMPLES) -> float:
    """
    Stochastic Monte Carlo expected cross section at nominal energy e_nom:
    - sample ISR fractions x
    - compute ISR-reduced energies E_eff_isr = sqrt((1-x) * e_nom^2)
    - sample Gaussian-smeared energies around E_eff_isr
    - evaluate BW resonances at smeared energies
    - average and add unsmeared Bhabha(E_nom)
    """
    if rng is None:
        rng = GLOBAL_RNG
    x_samp = sample_isr_x(e_nom**2, rng, n_samples)
    e_eff_isr = np.sqrt((1.0 - x_samp) * e_nom**2)
    e_smeared = rng.normal(loc=e_eff_isr, scale=SIGMA_COMBINED)
    sigma_jpsi = B_ACC * breit_wigner_sigma(e_smeared, M_JPSI, GAMMA_JPSI, GAMMAEE_JPSI)
    sigma_psip = B_ACC * breit_wigner_sigma(e_smeared, M_PSIP, GAMMA_PSIP, GAMMAEE_PSIP)
    return float(np.mean(sigma_jpsi + sigma_psip) + BHABHA_INTERP(e_nom))


def mc_deterministic_like(e_nom: float,
                          rng=None,
                          n_quad: int = DEFAULT_N_QUAD,
                          n_gauss_per_node: int = 200) -> float:
    """
    Deterministic-style MC: use same Gauss-Legendre quadrature nodes as theory,
    but for each node draw a small batch of Gaussian-smeared energies and average.
    This is useful to validate that stochastic MC converges to theory.
    """
    if rng is None:
        rng = GLOBAL_RNG

    # quadrature nodes on [0,1]
    x_nodes, w_nodes = np.polynomial.legendre.leggauss(n_quad)
    x_nodes = 0.5 * (x_nodes + 1.0)
    w_nodes = 0.5 * w_nodes

    s_nom = e_nom**2
    R_vals = isr_radiator_oalpha(x_nodes, s_nom)
    R_norm = np.sum(w_nodes * R_vals)
    if R_norm <= 0:
        R_weighted = w_nodes
    else:
        R_weighted = w_nodes * (R_vals / R_norm)

    accum = 0.0
    for x, w_eff in zip(x_nodes, R_weighted):
        s_eff = (1.0 - x) * s_nom
        if s_eff <= 0:
            continue
        e_eff = np.sqrt(s_eff)
        # sample a small Gaussian ensemble for this node
        Es = rng.normal(loc=e_eff, scale=SIGMA_COMBINED, size=n_gauss_per_node)
        sj = B_ACC * breit_wigner_sigma(Es, M_JPSI, GAMMA_JPSI, GAMMAEE_JPSI)
        sp = B_ACC * breit_wigner_sigma(Es, M_PSIP, GAMMA_PSIP, GAMMAEE_PSIP)
        accum += w_eff * np.mean(sj + sp)
    return float(accum + BHABHA_INTERP(e_nom))


# -----------------
# Deterministic theory (ISR convolution + Gaussian smearing)
# -----------------
def theory_with_isr(e_vals: np.ndarray,
                    n_quad: int = DEFAULT_N_QUAD,
                    show_progress: bool = True) -> np.ndarray:
    """
    For each energy E0 in e_vals:
      - compute normalized ISR weights R(x) on quadrature nodes
      - integrate Breit–Wigner contributions at ISR-reduced energies
      - after loop: Gaussian-smear the BW-only array and add unsmeared Bhabha
    """
    out = np.zeros_like(e_vals)
    x_nodes, w_nodes = np.polynomial.legendre.leggauss(n_quad)
    x_nodes = 0.5 * (x_nodes + 1.0)
    w_nodes = 0.5 * w_nodes

    start = time.time()
    total = len(e_vals)
    for idx, e0 in enumerate(e_vals, start=1):
        s_nom = e0**2
        R_vals = isr_radiator_oalpha(x_nodes, s_nom)
        R_norm = np.sum(w_nodes * R_vals)
        if R_norm <= 0:
            R_weighted = w_nodes
        else:
            R_weighted = w_nodes * (R_vals / R_norm)

        sigma_sum = 0.0
        for x, w_eff in zip(x_nodes, R_weighted):
            s_eff = (1.0 - x) * s_nom
            if s_eff <= 0:
                continue
            e_eff = np.sqrt(s_eff)
            sj = B_ACC * breit_wigner_sigma(e_eff, M_JPSI, GAMMA_JPSI, GAMMAEE_JPSI)
            sp = B_ACC * breit_wigner_sigma(e_eff, M_PSIP, GAMMA_PSIP, GAMMAEE_PSIP)
            sigma_sum += w_eff * (sj + sp)
        out[idx - 1] = sigma_sum

        if show_progress and (idx % 50 == 0 or idx == total):
            progress_bar(idx, total, start)

    out_smeared = smear_gaussian_fft(out, e_vals, SIGMA_COMBINED)
    return out_smeared + BHABHA_INTERP(e_vals)


# -----------------
# Progress bar helper
# -----------------
def progress_bar(current: int, total: int, start_time: float, bar_length: int = 40):
    fraction = current / total
    filled = int(bar_length * fraction)
    elapsed = time.time() - start_time
    sys.stdout.write(
        f"\r|{'█' * filled}{'-' * (bar_length - filled)}| "
        f"{fraction * 100:5.1f}% ({current}/{total}) Elapsed: {elapsed:.1f}s"
    )
    sys.stdout.flush()
    if current == total:
        print(f"\nDone. Total time: {elapsed:.1f}s")


# -----------------
# Convergence test
# -----------------
def convergence_test(e0: float,
                     n_samples_list=(1000, 5000, 20000, 100000),
                     rng=None):
    """Run MC at increasing n_samples and compare to theory_with_isr(e0)."""
    if rng is None:
        rng = GLOBAL_RNG

    print(f"\nConvergence test at E0 = {e0} GeV")
    # deterministic "truth"
    t0 = time.time()
    truth = theory_with_isr(np.array([e0]), n_quad=DEFAULT_N_QUAD, show_progress=False)[0]
    print(f"Deterministic theory_with_isr: {truth:.6e} nb (computed in {time.time()-t0:.2f}s)")
    results = []
    for n in n_samples_list:
        t = time.time()
        mc_val = mc_sigma_with_isr(e0, rng=rng, n_samples=n)
        results.append((n, mc_val, mc_val - truth, time.time() - t))
        print(f" MC n={n:8d} -> {mc_val:.6e} nb  Δ={mc_val-truth:+.6e}  time={time.time()-t:.2f}s")
    return truth, results


# -----------------
# Export utilities
# -----------------
def export_results(filename_prefix: str,
                   energies: np.ndarray,
                   sigma_obs: np.ndarray,
                   sigma_err: np.ndarray,
                   theory_noisr: np.ndarray,
                   theory_isr: np.ndarray):
    """Save CSV, NPZ, and send plot PNG to a results folder."""
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{filename_prefix}_data.csv")
    np.savetxt(csv_path,
               np.vstack([energies, sigma_obs, sigma_err]).T,
               delimiter=",",
               header="E_meas, sigma_obs(nb), sigma_err(nb)",
               fmt="%0.8e")
    np.savez_compressed(os.path.join(out_dir, f"{filename_prefix}_arrays.npz"),
                        E_meas=energies,
                        sigma_obs=sigma_obs,
                        sigma_err=sigma_err,
                        theory_noisr=theory_noisr,
                        theory_isr=theory_isr)

    print(f"Exported CSV -> {csv_path}")
    print(f"Exported NPZ -> {os.path.join(out_dir, f'{filename_prefix}_arrays.npz')}")


# -----------------
# Main CLI and run
# -----------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="J/ψ ISR + smearing simulation")
    parser.add_argument("--fast-test", action="store_true",
                        help="Run a fast test: fewer MC samples, fewer grid points.")
    parser.add_argument("--export", action="store_true",
                        help="Export data (CSV and NPZ) + save plot.")
    parser.add_argument("--n-mc", type=int, default=None,
                        help="Override number of MC samples for the simulation.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Don't display the plot (useful for batch runs).")
    parser.add_argument("--convergence", action="store_true",
                        help="Run convergence test and exit.")
    args = parser.parse_args(argv)

    # Adjust defaults for fast test
    global N_MC_SAMPLES, DEFAULT_N_QUAD
    if args.fast_test:
        print("Fast test mode: lowering sample/precision for quicker execution.")
        N_MC_SAMPLES = 5000
        DEFAULT_N_QUAD = 120
    if args.n_mc is not None:
        N_MC_SAMPLES = int(args.n_mc)
    else:
        N_MC_SAMPLES = DEFAULT_N_MC_SAMPLES

    # Bind the BHABHA_INTERP (already built globally) to local var for clarity
    global BHABHA_INTERP
    BHABHA_INTERP = build_bhabha_interpolator()

    # If requested, run convergence test and exit
    if args.convergence:
        e0_test = M_JPSI
        convergence_test(e0_test, n_samples_list=(1000, 5000, 20000, 100000))
        return

    # --- Monte Carlo scan ---
    print("Running Monte Carlo scan...")
    E_meas = []
    sigma_meas = []
    sigma_err = []

    mc_start = time.time()
    total_points = len(E_SCAN)
    for i, e0 in enumerate(E_SCAN, start=1):
        sigma_avg = mc_sigma_with_isr(e0, rng=GLOBAL_RNG, n_samples=N_MC_SAMPLES)
        n_expected = sigma_avg * L_INT * EFFICIENCY
        n_observed = GLOBAL_RNG.poisson(n_expected)
        sigma_observed = n_observed / (L_INT * EFFICIENCY)
        sigma_unc = np.sqrt(n_observed) / (L_INT * EFFICIENCY) if n_observed > 0 else 1.0 / (L_INT * EFFICIENCY)

        E_meas.append(e0)
        sigma_meas.append(sigma_observed)
        sigma_err.append(sigma_unc)

        progress_bar(i, total_points, mc_start)
    E_meas = np.array(E_meas)
    sigma_meas = np.array(sigma_meas)
    sigma_err = np.array(sigma_err)

    # --- Theory curves ---
    print("\nComputing theory curves...")
    E_vals = np.linspace(2.8, 3.8, DEFAULT_EGRID_POINTS if not args.fast_test else 2000)

    # No-ISR: smear BW with Gaussian, add unsmeared Bhabha
    bw_vals = B_ACC * breit_wigner_sigma(E_vals, M_JPSI, GAMMA_JPSI, GAMMAEE_JPSI) \
              + B_ACC * breit_wigner_sigma(E_vals, M_PSIP, GAMMA_PSIP, GAMMAEE_PSIP)
    sigma_noisr = BHABHA_INTERP(E_vals) + smear_gaussian_fft(bw_vals, E_vals, SIGMA_COMBINED)

    # With ISR: deterministic convolution (slow) with progress bar
    sigma_isr = theory_with_isr(E_vals, n_quad=DEFAULT_N_QUAD, show_progress=True)

    # --- Plot & export ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(E_meas, sigma_meas, yerr=sigma_err, xerr=SIGMA_COMBINED/np.sqrt(N_MC_SAMPLES),
                fmt="o", color="red", label="MC simulated data")
    ax.plot(E_vals, sigma_noisr, linestyle="--", color="black", label="Theory (no ISR)")
    ax.plot(E_vals, sigma_isr, linestyle="-", color="blue", label="Theory (with ISR)")
    ax.set_xlabel("CM Energy (GeV)")
    ax.set_ylabel("Cross section (nb)")
    ax.set_title("Simulated J/ψ and ψ(2S) cross section scan (with ISR)")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    if args.export:
        prefix = "jpsi_scan"
        export_results(prefix, E_meas, sigma_meas, sigma_err, sigma_noisr, sigma_isr)
        fig.savefig(os.path.join("results", f"{prefix}_plot.png"), dpi=200)
        print(f"Saved plot to results/{prefix}_plot.png")

    if not args.no_plot:
        plt.show()
    else:
        plt.close(fig)

    # optional quick validation: deterministic-like MC reproduction
    print("\nRunning deterministic-like MC for one central point (validation)...")
    e_val = M_JPSI
    det_like = mc_deterministic_like(e_val, rng=GLOBAL_RNG, n_quad=min(200, DEFAULT_N_QUAD), n_gauss_per_node=200)
    theory_at_e = theory_with_isr(np.array([e_val]), n_quad=DEFAULT_N_QUAD, show_progress=False)[0]
    print(f"deterministic-like MC @ {e_val:.4f} GeV: {det_like:.6e} nb")
    print(f"theory_with_isr                 : {theory_at_e:.6e} nb")
    print(f"difference (det_like - theory)  : {det_like - theory_at_e:+.6e} nb")

    print("\nDone.")

if __name__ == "__main__":
    main()
