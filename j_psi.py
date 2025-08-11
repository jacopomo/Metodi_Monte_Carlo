import numpy as np
from scipy import integrate, signal
import matplotlib.pyplot as plt

# --- constants ---
alpha = 1/137.035999084
GEV2_TO_NB = 389379.338

# -----------------
# Bhabha cross section (tree-level QED, massless) integrated over cosθ acceptance
# -----------------
def bhabha(Ecm, c_min=0.6):
    """
    Bhabha cross section for e⁺e⁻ → e⁺e⁻ at center-of-mass energy Ecm.
    Parameters:
    - Ecm: Center-of-mass energy (GeV)
    - c_min: Minimum cosine of the scattering angle (default 0.6)
    Returns:
    - Cross section in nb (nanobarns)
    """
    s = Ecm**2
    def integrand(c):
        """
        Integrand for the Bhabha cross section.
        Parameters:
        - c: Cosine of the scattering angle
        Returns:
        - Value of the integrand
        """
        t = -0.5 * s * (1 - c)
        u = -0.5 * s * (1 + c)
        return (s**2 + u**2)/(t**2) + (t**2 + u**2)/(s**2) + 2*u**2/(s*t)
    integral, _ = integrate.quad(integrand, -c_min, c_min, epsrel=1e-9)
    sigma_GeV2 = (np.pi * alpha**2 / s) * integral
    return sigma_GeV2 * GEV2_TO_NB

# -----------------
# Breit–Wigner cross section for J/ψ or ψ' → e⁺e⁻
# -----------------
def sigma_true(E, M=3.0969, Gamma=0.0000929, Gamma_ee=5.547e-6):
    """
    Breit-Wigner cross section for J/ψ or ψ' decaying to e⁺e⁻.
    Parameters:
    - E: Center-of-mass energy (GeV
    - M: Mass of the resonance (GeV, default J/ψ mass)
    - Gamma: Total width of the resonance (GeV, default J/ψ width)
    - Gamma_ee: Partial width to e⁺e⁻ (GeV, default J/ψ → e⁺e⁻ width)
    Returns:
    - Cross section in nb (nanobarns)
    """
    s = E**2
    num = 12 * np.pi * Gamma_ee**2
    denom = (s - M**2)**2 + (M**2) * (Gamma**2)
    return num / denom * GEV2_TO_NB

# -----------------
# Acceptance factor for angular cuts
# -----------------
def acceptance_factor(c_max=0.6):
    """
    Calculate the acceptance factor for the angular cuts in the Bhabha process.
    Parameters:
    - c_max: Maximum cosine of the scattering angle (default 0.6)
    Returns:
    - Acceptance factor (dimensionless)
    """
    num = integrate.quad(lambda u: 1 + u**2, -c_max, c_max)[0]
    den = integrate.quad(lambda u: 1 + u**2, -1, 1)[0]
    return num / den

b_acc = acceptance_factor(0.6)

# -----------------
# FFT smearing with Gaussian kernel
# -----------------
def smear_with_gaussian(y_vals, E_vals, sigma_E):
    """
    Apply Gaussian smearing to a set of values using FFT convolution.
    Parameters:
    - y_vals: Values to be smeared (cross section values)
    - E_vals: Corresponding energy values (GeV)
    - sigma_E: Standard deviation of the Gaussian kernel (GeV)
    Returns:
    - Smoothed values after convolution
    """
    dE = E_vals[1] - E_vals[0]
    kernel = np.exp(-0.5 * (np.arange(-len(y_vals)//2, len(y_vals)//2)*dE / sigma_E)**2)
    kernel /= np.sum(kernel)
    return signal.fftconvolve(y_vals, kernel, mode='same')

# -----------------
# Setup: theory curves
# -----------------
E_vals = np.linspace(2.8, 3.8, 2000)
sigma_bkg = np.array([bhabha(E) for E in E_vals])
sigma_jpsi = b_acc * sigma_true(E_vals)
sigma_psip = b_acc * sigma_true(E_vals, M=3.6861, Gamma=0.000294, Gamma_ee=2.33142e-6)

# The total cross section is the sum of the background and resonances (smeared with two Gaussians, one for the beam energy spread and one for the detector resolution)
sigma_smeared = sigma_bkg + smear_with_gaussian(sigma_jpsi + sigma_psip, E_vals, sigma_E=0.003*np.sqrt(2)) # detector resolution and beam energy spread both reported to be 3 MeV

# -----------------
# Real-world parameters
# -----------------
L_inst_cm2_s = 2e30  # example SPEAR instantaneous luminosity
L_inst = L_inst_cm2_s * 1e-33  # nb^-1 s^-1
efficiency = 1  # example detector efficiency (Mark I), if known; else use 1.0
M_jpsi = 3.0969
M_psip = 3.6861
sigma_E= 0.003  # beam energy spread (GeV)

# Scan strategy: more points near resonances
dEs = 0.018
E_scan = np.concatenate([
    np.linspace(2.8, M_jpsi - dEs, 6),
    np.linspace(M_jpsi - dEs, M_jpsi + dEs, 240),
    np.linspace(M_jpsi + dEs, M_psip - dEs, 8),
    np.linspace(M_psip - dEs, M_psip + dEs, 240),
    np.linspace(M_psip + dEs, 3.80, 4)
])
n_scan = len(E_scan)

# Total integrated luminosity per point
t_per_point = 100 * 3600  # 100 hours per point, adjust as needed
L_int = L_inst * t_per_point  # nb^-1


# -----------------
# Beam energy spread averaging
# -----------------
rng = np.random.default_rng(seed=64)

def beam_spread_average(E_nom, sigma_smeared, E_vals, sigma_beam=0.003, n_samples=1000, rng=None):
    """
    Average cross section over beam energy spread using Gaussian sampling.
    Parameters:
    - E_nom: Nominal energy (GeV)
    - sigma_smeared: Smeared cross section values (GeV^2)
    - E_vals: Energy values corresponding to sigma_smeared (GeV)
    - sigma_beam: Standard deviation of beam energy spread (GeV)
    - n_samples: Number of samples to average over
    - rng: Random number generator for reproducibility
    Returns:
    - Average cross section over the beam energy spread (nb)
    """

    # Sample E_true energies from Gaussian around E_nom
    E_samples = rng.normal(loc=E_nom, scale=sigma_beam, size=n_samples)

    # Interpolate sigma_smeared at sampled energies
    sigma_samples = np.interp(E_samples, E_vals, sigma_smeared, left=sigma_smeared[0], right=sigma_smeared[-1])


    # Return average cross section over samples
    return np.mean(sigma_samples)

# -----------------
# Simulate "true" experimental data
# -----------------

E_meas, sigma_meas, sigma_err = [], [], []

for E0 in E_scan:
    # Average over beam energy spread
    sigma_avg = beam_spread_average(E0, sigma_smeared, E_vals, sigma_beam=0, n_samples=1000, rng=rng)

    # Expected counts
    N_exp = sigma_avg * L_int * efficiency

    # Poisson fluctuation for observed counts
    N_obs = rng.poisson(N_exp)

    # Convert back to cross section and uncertainty
    sigma_obs = N_obs / (L_int * efficiency)
    sigma_unc = np.sqrt(N_obs) / (L_int * efficiency) if N_obs > 0 else 1 / (L_int * efficiency)

    E_meas.append(E0)
    sigma_meas.append(sigma_obs)
    sigma_err.append(sigma_unc)

E_meas = np.array(E_meas)
sigma_meas = np.array(sigma_meas)
sigma_err = np.array(sigma_err)

# -----------------
# Plot results
# -----------------
plt.figure(figsize=(10,6))
plt.plot(E_vals, sigma_smeared, label="Expected (smeared theory)", color="black", linestyle="--")
plt.errorbar(E_meas, sigma_meas, yerr=sigma_err, xerr=sigma_E ,fmt='o', label="Simulated data", color="red")
plt.xlabel("CM Energy (GeV)")
plt.ylabel("Cross section (nb)")
plt.title("Simulated energy scan (counts → cross section)")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("j_psi_simulated_scan.png")
plt.show()
