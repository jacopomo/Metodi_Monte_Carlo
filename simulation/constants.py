# jpsi/simulation/constants.py

"""
Physics constants, experimental apparatus parameters, and default parameters for the J/psi simulation.
All energies are in GeV, cross sections in nb, angles in radians unless otherwise specified.
"""

import numpy as np
from scipy import integrate

# Fundamental constants
alpha = 1 / 137.035999      # Fine-structure constant
gev2_to_nb = 389379.338     # Conversion factor from GeV^-2 to nb
m_e_gev = 0.00051099895     # Electron mass in GeV

# Resonance parameters (PDG 2024 values)
m_jpsi = 3.0969             # Mass of J/ψ in GeV
gamma_jpsi = 9.3e-5         # J/ψ total width in GeV
gamma_ee_jpsi = 5.55e-6     # J/ψ ee partial width in GeV

m_psi2s = 3.6861            # Mass of ψ(2S) in GeV
gamma_psi2s = 3.0e-4        # ψ(2S) total width in GeV
gamma_ee_psi2s = 2.35e-6    # ψ(2S) ee partial width in GeV

# Detector acceptance & resolution
cos_cut = 0.6               # |cos(theta)| < 0.6 acceptance
beam_resolution = 0.003     # Beam energy spread (Gaussian sigma) in GeV
detector_resolution = 0.003 # Detector energy resolution (Gaussian sigma) in GeV
energy_resolution = float(np.sqrt(beam_resolution**2 + detector_resolution**2))

def acceptance(dist_func=lambda u: 1 + u**2, cos_max: float = cos_cut, norm: bool = True) -> float:
    """
    Geometric acceptance for a given angular cut and angular distribution.

    Parameters
    ----------
    dist_func : callable
        Angular distribution function of cos(theta).
    cos_max : float
        Angular cut (default = cos_cut from constants).
    norm : bool
        Whether to normalize by total integral.
    
    Returns
    -------
    float
        Acceptance fraction.
    """
    num = integrate.quad(dist_func, -cos_max, cos_max, epsrel=1e-9)[0]
    den = integrate.quad(dist_func, -1, 1, epsrel=1e-11)[0] if norm else 1.0
    return num / den


# Luminosity/exposure (used for Poisson fluctuations)
l_instant_cm2 = 2e30                    # cm^-2 s^-1
l_instant_nb = l_instant_cm2 * 1e-33    # nb^-1 s^-1
t_per_point = 100 * 3600                # 100 hours per point in seconds
l_int = l_instant_nb * t_per_point      # Integrated luminosity per point in nb^-1
efficiency = 1.0

# Simulation defaults
n_mc = int(1e5)                                     # Number of MC samples per energy point
e_min = 2.8                                         # Minimum energy for scan (GeV)
e_max = 3.8                                         # Maximum energy for scan (GeV)
n_escan_points = int(1e4)                           # Number of energy grid points for cross section calculations
x_grid = np.logspace(-8, np.log10(0.999), 50000)    # ISR x grid for PDF/CDF

# Heterogeneous energy scan for data simulation
padding = 0.022  
mc_energies = np.concatenate([
    np.linspace(e_min, m_jpsi - padding, 12),                 # below J/ψ
    np.linspace(m_jpsi - padding, m_jpsi + padding, 50),      # around J/ψ
    np.linspace(m_jpsi + padding, m_psi2s - padding, 16),     # between resonances
    np.linspace(m_psi2s - padding, m_psi2s + padding, 20),    # around ψ(2S)
    np.linspace(m_psi2s + padding, e_max, 6)                  # above ψ(2S)
])

global_rng = np.random.default_rng(12345)

isr_on = True      # Whether to include ISR effects in MC simulation