import numpy as np
from scipy import integrate, signal
import matplotlib.pyplot as plt
import sys
import time

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
# Real-world parameters
# -----------------
sigma_beam = 0.003
sigma_detector = 0.003
sigma_combined = np.sqrt(sigma_beam**2 + sigma_detector**2)
L_inst_cm2_s = 2e30 # instantaneous luminosity in cm^-2 s^-1
L_inst = L_inst_cm2_s * 1e-33  # nb^-1 s^-1
t_per_point = 100 * 3600  # 100 hours
L_int = L_inst * t_per_point
efficiency = 1.0 # detector efficiency (100% for simplicity)

# -----------------
# Energy scan parameters
# -----------------
dEs = 0.018
E_scan = np.concatenate([
    np.linspace(2.8, 3.0969 - dEs, 6),
    np.linspace(3.0969 - dEs, 3.0969 + dEs, 240),
    np.linspace(3.0969 + dEs, 3.6861 - dEs, 8),
    np.linspace(3.6861 - dEs, 3.6861 + dEs, 240),
    np.linspace(3.6861 + dEs, 3.80, 4)
])

# -----------------
# MC data generation
# -----------------
rng = np.random.default_rng(seed=12345)
n_mc_samples = 5000

def mc_smeared_sigma(E_nom):
    """
    Monte Carlo simulation to average cross section over beam energy spread.
    Parameters:
    - E_nom: Nominal center-of-mass energy (GeV)
    Returns:
    - Average cross section after smearing (nb)
    """
    # Sample energies from combined Gaussian around E_nom
    E_samples = rng.normal(loc=E_nom, scale=sigma_combined, size=n_mc_samples)
    
    # Compute total cross section (BW J/psi + BW psi' + Bhabha background) at each sampled energy
    sigma_jpsi = b_acc * sigma_true(E_samples)
    sigma_psip = b_acc * sigma_true(E_samples, M=3.6861, Gamma=0.000294, Gamma_ee=2.33142e-6)
    sigma_bkg = np.array([bhabha(E) for E in E_samples])
    
    sigma_tot_samples = sigma_jpsi + sigma_psip + sigma_bkg
    
    # Average over MC samples to approximate convolution
    return np.mean(sigma_tot_samples)

# -----------------
# Progress bar for MC loop
# -----------------
def progress_bar(current, total, start_time, bar_length=40):
    """
    Display a progress bar in the terminal.

    Parameters:
    - current: current progress (int)
    - total: total iterations (int)
    - bar_length: length of the progress bar (int)
    """
    fraction = current / total
    filled_length = int(bar_length * fraction)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = fraction * 100
    elapsed = time.time() - start_time
    sys.stdout.write(f'\r|{bar}| {percent:.1f}% ({current}/{total}) Elapsed: {elapsed:.1f}s')
    sys.stdout.flush()
    if current == total:
        print(f'\nDone! Total time: {elapsed:.1f} seconds.')


# -----------------
# Main MC loop
# -----------------
E_meas = []
sigma_meas = []
sigma_err = []
start = time.time()

for i, E0 in enumerate(E_scan, 1):
    sigma_avg = mc_smeared_sigma(E0)
    
    # Expected counts for integrated luminosity and efficiency
    N_exp = sigma_avg * L_int * efficiency
    
    # Poisson fluctuation in observed counts
    N_obs = rng.poisson(N_exp)
    
    # Convert counts back to cross section and uncertainty
    sigma_obs = N_obs / (L_int * efficiency)
    sigma_unc = np.sqrt(N_obs) / (L_int * efficiency) if N_obs > 0 else 1/(L_int * efficiency)
    
    E_meas.append(E0)
    sigma_meas.append(sigma_obs)
    sigma_err.append(sigma_unc)

    progress_bar(i, len(E_scan), start)
E_meas = np.array(E_meas)
sigma_meas = np.array(sigma_meas)
sigma_err = np.array(sigma_err)


# -----------------
# FFT smearing with Gaussian kernel for expected counts
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
E_vals = np.linspace(2.8, 3.8, 50000)
sigma_bkg = np.array([bhabha(E) for E in E_vals])
sigma_jpsi = b_acc * sigma_true(E_vals)
sigma_psip = b_acc * sigma_true(E_vals, M=3.6861, Gamma=0.000294, Gamma_ee=2.33142e-6)

# The total cross section is the sum of the background and resonances (smeared with two Gaussians, one for the beam energy spread and one for the detector resolution)
sigma_smeared = sigma_bkg + smear_with_gaussian(sigma_jpsi + sigma_psip, E_vals, sigma_E=sigma_combined) 

# Plot simulated data points
plt.figure(figsize=(10,6))
plt.errorbar(E_meas, sigma_meas, yerr=sigma_err, xerr=sigma_combined, fmt='o', label='MC simulated data (pure MC smearing)', color='red')
plt.plot(E_vals, sigma_smeared, label="Expected (smeared theory)", color="black", linestyle="--")
plt.legend()
plt.xlabel("CM Energy (GeV)")
plt.ylabel("Cross section (nb)")
plt.title("Simulated J/ψ and ψ' Cross Section Scan")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("j_psi_simulated_scan.png")
plt.show()
