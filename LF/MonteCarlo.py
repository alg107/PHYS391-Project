"""
Monte-Carlo method using Iso.py Isochrone library
Author: Alex Goodenbour
Version: Outlier Fixed
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import lib.IMF as IMF
from progressbar import progressbar as pb
import pandas as pd
from lib.Iso import Isochrone
# 1: Constant and Helper Function Definition


# The number of samples of metallicity and IMF
N = int(1e7)
#N = int(1e9)



# Metallicity

metallicity_mean = 0.0
metallicity_std = 0.4
metallicity_bins = 39

# IMF

IMF_bins = 100

# 2: Sampling
# 2.1: Metallicity

# Sampling metallicities from a normal distribution
sampled_metallicities = np.random.normal(metallicity_mean, metallicity_std, N)

plt.figure()
plt.title("Metallicity Distribution")
#metallicity_bins_seq = np.linspace(-2.279, 0.198, 39)
plt.hist(sampled_metallicities, 100, density=True, histtype='step')
X = np.linspace(-2*metallicity_std, 2*metallicity_std, 10000)
plt.plot(X, ss.norm.pdf(X, metallicity_mean, metallicity_std))

print("Metallicity Done")

# 2.2: IMF

# Sampling masses from lognormal IMF
sampled_IMF = IMF.IMF_sample(N)
##np.save("sampledIMF_N"+str(N), np.array(sampled_IMF))

#sampled_IMF = IMF.load_samples()


plt.figure()
plt.title("IMF Distribution")
IMF_seq = np.linspace(0.8, 1.1, IMF_bins)
plt.hist(sampled_IMF, IMF_seq, density=True, histtype='step')
X = np.linspace(0.8, 1.1, 1000)
plt.plot(X, IMF.chabrier2_V(X))

print("Masses Done")

# 3: Isochrones

def plot_samples(sampls, bins, title):
    plt.figure()

    # Plotting the LF histogram
    final_bins = bins
    bins_seq = np.linspace(-3.5, 1.0, final_bins)
    plt.hist(sampls, bins_seq)

    plt.title(title)
    plt.xlabel("Magnitude")
    plt.ylabel("Count")

total_sampled_mags = []

for t in [1,2,3]:
    print("Interpolating Branch", t)
    iso = Isochrone(typs=[t])

    # This is the array in which we will build up our final histogram
    # Cutting the isochrone into constant metallicity slices
    sampled_mags = []
    for i, m in pb(enumerate(sampled_IMF), max_value=N):
        val = iso.interpolate(m, sampled_metallicities[i])
        if not np.isnan(val):
            sampled_mags.append(val)
    sampled_mags = np.array(sampled_mags)

    np.save("Results/MCSamples/type"+str(t)+"N"+str(N), sampled_mags)
    print("Efficiency: ", len(sampled_mags)*100/N, "%")

    plot_samples(sampled_mags, 75, "Type "+str(t)+" LF")
    total_sampled_mags.extend(sampled_mags)

plot_samples(total_sampled_mags, 75, "Total LF")
plt.show()
