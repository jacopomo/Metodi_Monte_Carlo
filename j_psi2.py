import numpy as np
from scipy import integrate, signal
import matplotlib.pyplot as plt

# --- constants ---
alpha = 1/137.035999084
GEV2_TO_NB = 389379.338

# -----------------
# Exact Bhabha cross section (massless) integrated over cosθ acceptance
# -----------------
def bhabha(Ecm, c_min=0.6):
    """
    Integrated Bhabha scattering cross section (tree-level, massless).
    Ecm: CM energy in GeV
    c_min: minimum cos(theta) for acceptance
    Returns cross section in nb.
    """
    s = Ecm**2
    def integrand(c):
        t = -s/2.0 * (1.0 - c)
        u = -s/2.0 * (1.0 + c)
        term1 = (s**2 + u**2) / (t**2)
        term2 = (t**2 + u**2) / (s**2)
        term3 = (2.0 * u**2) / (s * t)
        return term1 + term2 + term3

    integral, _ = integrate.quad(integrand, -c_min, c_min, epsrel=1e-9)
    sigma_GeV2 = (np.pi * alpha**2 / s) * integral
    return sigma_GeV2 * GEV2_TO_NB

# -----------------
# Breit–Wigner true resonance cross section for e+e- → J/psi → e+e-
# -----------------
def sigma_true(E, M=3.0969, Gamma=0.0000929, Gamma_ee=5.547e-6):
    """
    True cross section for J/psi production in e+e- collisions.
    E: CM energy in GeV
    M: mass of the J/psi particle in GeV
    Gamma: total width of the J/psi particle in GeV
    Gamma_ee: decay width of J/psi to e+e- in GeV
    Returns the cross section in nb.
    """
    s = E**2
    M2 = M**2
    Gamma2 = Gamma**2
    num = 12 * np.pi * (Gamma_ee**2)
    denom = (s - M2)**2 + M2 * Gamma2
    return num / denom * GEV2_TO_NB

# -----------------
# Acceptance factor for angular cuts (resonance angular dist ~ 1+cos²θ)
# -----------------
def acceptance_factor(c_max=0.6):
    """
    Calculate the acceptance factor for the angular distribution of the resonance.
    c_max: maximum cos(theta) for acceptance
    Returns the acceptance factor.
    """
    num = integrate.quad(lambda u: 1.0 + u**2, -c_max, c_max)[0]
    den = integrate.quad(lambda u: 1.0 + u**2, -1.0, 1.0)[0]
    return num / den

b_acc = acceptance_factor(0.6)

# -----------------
# FFT Gaussian smearing (convolution)
# -----------------
def smear_with_gaussian(y_vals, E_vals, sigma_E):
    """
    Smear the input y_vals using a Gaussian kernel based on the energy values E_vals.
    y_vals: array of cross section values to be smeared
    E_vals: array of CM energy values corresponding to y_vals
    sigma_E: standard deviation of the Gaussian kernel in GeV
    Returns the smeared y_vals.
    """
    # Construct Gaussian kernel on the same energy grid
    dE = E_vals[1] - E_vals[0]
    kernel_E = np.arange(-len(E_vals)//2, len(E_vals)//2) * dE
    gauss_kernel = np.exp(-0.5*(kernel_E/sigma_E)**2)
    gauss_kernel /= gauss_kernel.sum()  # normalize area = 1

    # FFT convolution
    y_smeared = signal.fftconvolve(y_vals, gauss_kernel, mode='same')
    return y_smeared

# -----------------
# Main: compute background + signal to obtain expected cross section
# -----------------
E_start, E_end, n_points = 2.8, 3.8, 2000
E_vals = np.linspace(E_start, E_end, n_points)
sigma_E = 0.003  # GeV, beam spread

# Compute background (Bhabha) curve
sigma_b_vals = np.array([bhabha(E) for E in E_vals])

# Compute unsmeared resonance cross section (with acceptance factor)
sigma_res1_vals = b_acc * sigma_true(E_vals)  # J/psi
sigma_res2_vals = b_acc * sigma_true(E_vals, M=3.6861, Gamma=0.000294, Gamma_ee=2.33142e-6)  # psi(2S)

# Smear resonance with beam energy spread (3 MeV)
sigma_res1_smeared = smear_with_gaussian(sigma_res1_vals, E_vals, sigma_E)
sigma_res2_smeared = smear_with_gaussian(sigma_res2_vals, E_vals, sigma_E)

# Add background and signal
sigma_total = sigma_b_vals + sigma_res1_smeared + sigma_res2_smeared


# -----------------
# Simulate measured data points with non-uniform scanning density and variable run time
# -----------------

# Machine instantaneous luminosity: 2e30 cm^-2 s^-1 -> in nb^-1 s^-1
L_int = 5e-3 * 3600*500 # nb^-1 in 500 hours

# Scan strategy: denser near resonances
M_jpsi = 3.0969
M_psip = 3.6861


# Regions: allocate more points near resonances
dEs = 0.03
E_far_left = np.linspace(2.8, M_jpsi-dEs, 4)
E_near_jpsi = np.linspace(M_jpsi-dEs, M_jpsi+dEs, 20)
E_between = np.linspace(M_jpsi+dEs, M_psip-dEs, 6)
E_near_psip = np.linspace(M_psip-dEs, M_psip+dEs, 20)
E_far_right = np.linspace(M_psip+dEs, 3.80, 2)

E_scan = np.concatenate([E_far_left, E_near_jpsi, E_between, E_near_psip, E_far_right])
n_scan = len(E_scan)

# Prepare measured data arrays
sigma_meas = []
sigma_err = []
E_meas = []
counts_meas = []

rng = np.random.default_rng(seed=42)  # reproducible randomness

for E0 in E_scan:
    # Simulate beam-energy spread for this point (draw true collision energy)
    E_true = np.random.normal(E0, sigma_E)

    # Theoretical cross section at E_true (interpolate smooth expected curve)
    sigma_th = np.interp(E_true, E_vals, sigma_total)  # in nb

    # Expected number of events (counts) at this point
    N_exp = sigma_th * L_int

    # Poisson fluctuation -> observed counts
    N_obs = np.random.poisson(N_exp)

    # Back to measured cross section and error (statistical)
    if N_obs > 0:
        sigma_observed = N_obs / L_int
        sigma_unc = np.sqrt(N_obs) / L_int
    else:
        sigma_observed = 0.0
        # approximate 1-sigma upper-limits style error for zero counts:
        sigma_unc = 1.0 / L_int

    # Save
    E_meas.append(E0)
    sigma_meas.append(sigma_observed)
    sigma_err.append(sigma_unc)
    counts_meas.append(N_obs)

E_meas = np.array(E_meas)
sigma_meas = np.array(sigma_meas)
sigma_err = np.array(sigma_err)
counts_meas = np.array(counts_meas)


# -----------------
# Plot results
# -----------------
plt.figure(figsize=(10,6))
plt.plot(E_vals, sigma_total, label="Expected (background + signal)", color="C0", linestyle='--', alpha=0.9)
plt.errorbar(E_meas, sigma_meas, xerr=sigma_E, yerr=sigma_err, fmt='o', color="C3", label="Simulated points (scan)", markersize=6, capsize=3)
plt.axvline(M_jpsi, color='k', linestyle=':', label=f"J/ψ ({M_jpsi} GeV)")
plt.axvline(M_psip, color='k', linestyle='--', label=f"ψ' ({M_psip} GeV)")
plt.xlabel("CM Energy (GeV)")
plt.ylabel("Cross section (nb)")
plt.title("Simulated energy scan near J/ψ and ψ' (variable density & run-time)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("j_psi2_scan_simulation.png")
plt.show()