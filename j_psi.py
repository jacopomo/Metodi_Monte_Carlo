import numpy as np
import scipy.integrate as integrate 
import matplotlib.pyplot as plt
# j_psi.py

alpha = 1/137.035999084  # Fine-structure constant

def bhabha(Ecm, c_min=0.6):
    """
    Integrated Bhabha scattering cross section (tree-level, massless).
    Ecm: CM energy in GeV
    theta_min: minimum polar angle in radians (detector geometry)
    Returns cross section in nb.
    """
    s = Ecm**2  # Mandelstam s

    def bhabha_anglular(c):
        """
        Integrand for the Bhabha scattering cross section.
        c: cos(theta), where theta is the scattering angle.
        Returns the integrand value.
        """
        t = -s/2 * (1 - c)
        u = -s/2 * (1 + c)
        I = (u**2 * (1/s + 1/t)**2) + (t/s)**2 + (s/t)**2
        return I

    # integrate over theta in the range
    integral, _ = integrate.quad(bhabha_anglular, -c_min, c_min)

    # multiply prefactor, convert to nb
    sigma = np.pi * alpha**2 * integral / s  # in GeV^-2
    return sigma * 389379.338  # in nb



# J/psi production cross section in e+e- collisions
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
    num = 12*np.pi * (Gamma_ee**2)
    denom = (s - M2)**2 + M2*Gamma2
    # convert GeV^-2 to nb
    return num / denom * 389379.338  # in nb

# Gaussian beam spread function
def beam_gaussian(E_prime, E0, resolution):
    """
    Gaussian beam spread function.
    E_prime: CM energy in GeV
    E0: nominal CM energy in GeV
    resolution: standard deviation of the Gaussian beam spread in GeV
    Returns the Gaussian value at E_prime.
    """
    return np.exp(-0.5*((E_prime - E0)/resolution)**2) / (resolution*np.sqrt(2*np.pi))

# Convolution: observed cross section at nominal E0
def sigma_obs(E0, resolution, c_min=0.6):
    """
    Observed cross section for J/psi production in e+e- collisions,
    taking into account the Gaussian beam spread.
    E0: nominal CM energy in GeV
    resolution: standard deviation of the Gaussian beam spread in GeV
    Returns the observed cross section in nb.
    """

    def resonance_angular(c):
        """
        Integrand for the resonance cross section.
        c: cos(theta), where theta is the scattering angle.
        Returns the integrand value.
        """
        return 1+c**2

    # Remember that the detector only accepts events with cos(theta) > c_min
    b = integrate.quad(resonance_angular, -c_min, c_min)[0] / integrate.quad(resonance_angular, -1, 1)[0] 

    integrand = lambda E_prime: b * sigma_true(E_prime) * beam_gaussian(E_prime, E0, resolution)
    result, _ = integrate.quad(integrand, 0, np.inf)  # integrate over CM energies
    return result

background = []
signal = []
resolution = 0.003  # resolution in GeV
n_points = 1000  # number of points for the background
E_start = 2.6  # starting CM energy in GeV
E_end = 8.0  # ending CM energy in GeV
Ecm_values = np.linspace(E_start, E_end, n_points) + np.random.normal(0, resolution) # CM energy from 2.6 to 8 GeV, with small noise given from SPEAR documentation
for i in range(n_points):
    Ecm = Ecm_values[i]
    background.append(bhabha(Ecm))
    signal.append(sigma_obs(Ecm, resolution))

plt.errorbar(np.linspace(E_start, E_end,n_points), background, yerr=0, xerr= resolution, label='Bhabha', color='blue', fmt=".")
plt.errorbar(np.linspace(E_start, E_end,n_points), signal, yerr=0, xerr= resolution, label='J/psi', color='red', fmt=".")
plt.xlabel('CM Energy (GeV)')
plt.ylabel('Cross Section (nb)')
plt.xscale('linear')
plt.yscale('log')
plt.title('Bhabha Scattering Cross Section')
plt.legend()
plt.grid()
plt.savefig('bhabha_scattering.png')
plt.show()