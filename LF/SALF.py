import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, splev
from scipy.integrate import simps, trapz
from scipy.ndimage.filters import gaussian_filter1d
from progressbar import progressbar as pb
from lib.Iso import Isochrone
from lib.IMF import chabrier2 as chabrier

iso = Isochrone()


# Normal distribution
def norm(x, m, s):
    return (1/np.sqrt(2*np.pi*s**2))*np.exp(-((x-m)**2/(2*s**2)))

# Mass distr. fn.
def MDF(z):
    return norm(z, 0.0, 0.4)

def phi(M, z, typs):
    ms, derivs = iso.inverse_interpolate(M, z, typs)
    phi_c = 0
    for i, m in enumerate(ms):
        phi_c += chabrier(m)*np.abs(derivs[i])
    return phi_c

RES = 1000
x = np.linspace(-3.5, 1.0, RES)

def Phi(M, typ):
    #zs = iso.zs
    zs = np.linspace(-100.0, 100.0, 1000) # This was where the bug was
    phis = np.array([phi(M, z, [typ]) for z in zs]) 
    MDFs = np.array([MDF(z) for z in zs])
    ys = phis*MDFs
    I = trapz(ys, zs)
    return I


plt.figure()
plt.xlabel("Magnitude")
plt.ylabel("Luminosity Function (Arbitrary Units)")

for i in pb([1,2,3], redirect_stdout=True):
    # Smoothing and plotting
    ys = np.array([Phi(M, i) for M in x])
    np.save("Results/SALF/xs_t"+str(i), x)
    np.save("Results/SALF/ys_t"+str(i), ys)
    plt.plot(x, ys)
    print("Done Branch ", i)


plt.show()

