import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d, make_interp_spline, BSpline, NearestNDInterpolator
import scipy.stats as ss
import IMF
from progressbar import progressbar as pb
import pandas as pd
from Iso import Isochrone
# 1: Constant and Helper Function Definition

bulge_age = 10 #Gyr

# The number of samples of metallicity and IMF
N = int(1e6)


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
#sampled_IMF = IMF.IMF_sample(N)
#np.save("sampledIMF_N"+str(N), np.array(sampled_IMF))

sampled_IMF = IMF.load_samples()


plt.figure()
plt.title("IMF Distribution")
IMF_seq = np.linspace(0, 2, IMF_bins)
plt.hist(sampled_IMF, IMF_seq, density=True, histtype='step')
X = np.linspace(0.001, 2, 10000)
plt.plot(X, IMF.chabrier(X))

print("Masses Done")

# 3: Isochrones

iso = Isochrone()

# First attempt at generating a LF

# Limits on metallicity from isochrone
plt.figure()

# This is the array in which we will build up our final histogram
# Cutting the isochrone into constant metallicity slices

sampled_mags = [iso.interpolate(m, sampled_metallicities[i]) for i, m in sampled_IMF]
sampled_mags = np.array(sampled_mags)

plt.figure()

# Plotting the LF histogram
final_bins = 200
bins_seq = np.linspace(-3.5, 1.0, final_bins)
plt.hist(sampled_mags, bins_seq, density=True, histtype='step')
counts,bin_edges = np.histogram(sampled_mags, bins_seq, density=True)

bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
err = np.random.rand(bin_centres.size)*100

plt.title("LF Histogram")
plt.xlabel("Magnitude")
plt.ylabel("Density")

plt.figure()
plt.scatter(bin_centres, counts, marker="x", color="black")

# # Creating a nice spline to see the pattern
# xnew = np.linspace(bin_centres.min(), bin_centres.max(), 1000) 
# spl = make_interp_spline(bin_centres, counts, k=3)  # type: BSpline
# power_smooth = spl(xnew)
# plt.plot(xnew, power_smooth)

plt.title("LF Scatter")
plt.xlabel("Magnitude")
plt.ylabel("Density")

print("LF attempt Done")
plt.show()
