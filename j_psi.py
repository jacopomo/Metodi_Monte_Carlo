import numpy as np

alpha = 1/137  # Fine-structure constant

def bhabha(Ecm):
    sigma = 389379 * np.pi * alpha**2 * 10.2071633672 / Ecm**2 #Ecm in GeV, sigma in nb
    #the first magic number is Gev-2 to nb and the second is the result of the theta integration; see desmos
    return sigma

a = bhabha(3.1)
print(a)