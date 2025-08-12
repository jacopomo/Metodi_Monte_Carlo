import numpy as np
from scipy import integrate, signal
import matplotlib.pyplot as plt
import sys
import time

# --- constants ---
alpha = 1/137.035999084
GEV2_TO_NB = 389379.338
m_e_GeV = 0.000511  # electron mass in GeV (for ISR beta)

# -----------------
# Bhabha cross section (tree-level QED, massless) integrated over cosθ acceptance
# -----------------
def bhabha(Ecm, c_min=0.6):
    s = Ecm**2
    def integrand(c):
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
    s = E**2
    num = 12 * np.pi * Gamma_ee**2
    denom = (s - M**2)**2 + (M**2) * (Gamma**2)
    return num / denom * GEV2_TO_NB

# -----------------
# Acceptance factor for angular cuts
# -----------------
def acceptance_factor(c_max=0.6):
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
    np.linspace(2.8, 3.0969 - dEs, 0),
    np.linspace(3.0969 - dEs, 3.0969 + dEs, 120),
    np.linspace(3.0969 + dEs, 3.6861 - dEs, 0),
    np.linspace(3.6861 - dEs, 3.6861 + dEs, 120),
    np.linspace(3.6861 + dEs, 3.80, 0)
])

# -----------------
# ISR model (leading-log radiator) utilities
# -----------------
def compute_beta(E_cm):
    """Compute beta for radiator f(x) = beta * x^{beta-1} (leading-log approx)."""
    s = E_cm**2
    # Protect against tiny s
    val = (2 * alpha / np.pi) * (np.log(s / (m_e_GeV**2)) - 1.0)
    # If numeric issues occur, bound beta to a small positive number:
    if val <= 0:
        return 1e-8
    return val

def sample_isr_x(E_cm, n_samples, rng):
    """
    Sample n_samples values of x from f(x)=beta * x^{beta-1} on (0,1).
    Uses inverse CDF: x = u^(1/beta).
    """
    beta = compute_beta(E_cm)
    u = rng.random(n_samples)
    # inverse CDF
    x = u ** (1.0 / beta)
    # numerical safety: ensure x in (0,1)
    x = np.clip(x, 0.0, 1.0 - 1e-15)
    return x

def sample_smeared_energies_with_isr(E_nom, n_samples, rng, sigma_beam=sigma_beam, sigma_detector=sigma_detector):
    """
    Return n_samples final energies after applying ISR (leading-log) and Gaussian smearing.
    ISR reduces energy: E_after_isr = E_nom * sqrt(1 - x).
    Then Gaussian smearing (combined) is applied around that energy.
    """
    # ISR sampling
    x = sample_isr_x(E_nom, n_samples, rng)
    E_after_isr = E_nom * np.sqrt(1.0 - x)  # shape (n_samples,)
    # apply Gaussian smearing (beam + detector combined)
    sigma_comb = np.sqrt(sigma_beam**2 + sigma_detector**2)
    E_final = rng.normal(loc=E_after_isr, scale=sigma_comb)
    return E_final

# -----------------
# MC data generation with ISR + smearing
# -----------------
rng = np.random.default_rng(seed=12345)
n_mc_samples = 10000  # 10000 for good statistics

def mc_smeared_sigma_with_isr(E_nom):
    """
    Monte Carlo average including ISR and Gaussian smearing.
    Returns average cross section in nb at nominal energy E_nom.
    """
    # Sample final energies after ISR and smearing
    E_samples = sample_smeared_energies_with_isr(E_nom, n_mc_samples, rng, sigma_beam, sigma_detector)
    # Evaluate components at these sampled energies
    sigma_jpsi = b_acc * sigma_true(E_samples)  # vectorized
    sigma_psip = b_acc * sigma_true(E_samples, M=3.6861, Gamma=0.000294, Gamma_ee=2.33142e-6)
    # bhabha is scalar-only -> evaluate per sample (could be optimized)
    sigma_bkg = np.array([bhabha(E) for E in E_samples])
    sigma_tot_samples = sigma_jpsi + sigma_psip + sigma_bkg
    return np.mean(sigma_tot_samples)

# -----------------
# Progress bar for MC loop
# -----------------
def progress_bar(current, total, start_time, bar_length=40):
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
# Main MC loop (ISR + smearing)
# -----------------
E_meas = []
sigma_meas = []
sigma_err = []
start = time.time()

for i, E0 in enumerate(E_scan, 1):
    sigma_avg = mc_smeared_sigma_with_isr(E0)
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
# FFT smearing with Gaussian kernel for expected counts (for comparison)
# -----------------
def smear_with_gaussian(y_vals, E_vals, sigma_E):
    dE = E_vals[1] - E_vals[0]
    kernel = np.exp(-0.5 * (np.arange(-len(y_vals)//2, len(y_vals)//2)*dE / sigma_E)**2)
    kernel /= np.sum(kernel)
    return signal.fftconvolve(y_vals, kernel, mode='same')

# -----------------
# Setup: theory curves on a fine grid for comparison
# -----------------
E_vals = np.linspace(2.8, 3.8, 100000) # high resolution
sigma_bkg = np.array([bhabha(E) for E in E_vals])
sigma_jpsi = b_acc * sigma_true(E_vals)
sigma_psip = b_acc * sigma_true(E_vals, M=3.6861, Gamma=0.000294, Gamma_ee=2.33142e-6)
sigma_smeared = sigma_bkg + smear_with_gaussian(sigma_jpsi + sigma_psip, E_vals, sigma_E=sigma_combined)

# -----------------
# Plot simulated data points vs FFT-smeared expected curve
# -----------------
plt.figure(figsize=(10,6))
plt.errorbar(E_meas, sigma_meas, yerr=sigma_err, xerr=sigma_combined, fmt='o', label='MC simulated data (ISR + smearing)', color='red')
plt.plot(E_vals, sigma_smeared, label="Expected (FFT double-smear)", color="black", linestyle="--")
plt.legend()
plt.xlabel("CM Energy (GeV)")
plt.ylabel("Cross section (nb)")
plt.title("Simulated J/ψ and ψ' Cross Section Scan (with ISR + smearing)")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("j_psi_simulated_scan_with_isr.png")
plt.show()
