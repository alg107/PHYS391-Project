import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

# 1: Constant and Helper Function Definition

bulge_age = 10 #Gyr
N = int(1e3)

# Finds nearest value in an array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Classifies stage based on statement in paper 1
def classify_stage(val):
    # 1: Red Giant
    # 2: RC
    # 3: Asymptotic Giant
    if val<=3:
        return 1
    elif val <= 6:
        return 2
    else:
        return 3 

# Gets a colour given a number from 1-3
def colour_from_type(typ):
    if typ==1:
        return "red"
    elif typ==2:
        return "blue"
    elif typ==3:
        return "green"
    else:
        return "yellow"

# Metallicity

metallicity_mean = 0.0
metallicity_std = 0.4
metallicity_bins = 39

# IMF

IMF_mean = 0.079
IMF_std = 0.69
IMF_bins = 39

# 2: Sampling
# 2.1: Metallicity

# Sampling metallicities from a normal distribution
sampled_metallicities = np.random.normal(metallicity_mean, metallicity_std, N)

plt.figure()
plt.title("Metallicity Distribution")
metallicity_bins_seq = np.linspace(-2.279, 0.198, 39)
plt.hist(sampled_metallicities, metallicity_bins_seq, density=True, histtype='step')

# 2.2: IMF

# Sampling masses from lognormal IMF
# Does it matter that this has natural log and the log in Chabrier is unspecified?
sampled_IMF = np.random.lognormal(IMF_mean, IMF_std, N)

plt.figure()
plt.title("IMF Distribution")
plt.hist(sampled_IMF, IMF_bins, density=True, histtype='step')

# 3: Isochrones

MM = np.column_stack((sampled_metallicities, sampled_IMF))

# Taking the useful stuff from the isochrone table
iso_table = np.loadtxt("iso.db")
MH = iso_table[:,1]
masses = iso_table[:,5]
Kmag = iso_table[:,32]
types = iso_table[:,9]

# Plotting these 3 vars in a box just to get a feel for the data
fig = plt.figure()
ax = Axes3D(fig)
plt.title("Isochrone mass-magnitude")
ax.scatter(masses, Kmag, MH, marker=".")
ax.set_xlabel("mass")
ax.set_ylabel("mag")
ax.set_zlabel("metallicity")


# First attempt at generating a LF

# Limits on metallicity from isochrone
metal_inf = np.min(MH)
metal_sup = np.max(MH)
print(metal_inf, metal_sup)
histplots = []

plt.figure()
# Doing each type individually
for type_ in [1,2,3]:
    sampled_mags = []
    # Cutting the isochrone into constant metallicity slices
    for i, MHval in enumerate(sampled_metallicities):
        nearestMH = find_nearest(MH, MHval)
        mags_i = []
        mass_i = []
        types_i = []
        for j, val in enumerate(MH):
            if val==nearestMH:
                mags_i.append(Kmag[j])
                mass_i.append(masses[j])
                types_i.append(classify_stage(types[j]))
        mags_i = np.array(mags_i)
        mass_i = np.array(mass_i)
        types_i = np.array(types_i)

        # Puts these all together in a big array
        data = np.column_stack((mass_i, mags_i, types_i))

        # Isolating the different types
        redgiants = data[data[:,2] == 1] 
        RC = data[data[:,2] == 2] 
        asymp = data[data[:,2] == 3] 

        # The one we're going to be using
        pnts = data[data[:,2] == type_] 


        # Using a linear spline to connect fixed points
        x = np.linspace(np.min(pnts[:,0]),np.max(pnts[:,0]),1000)
        spl = interp1d(pnts[:,0], pnts[:,1])

        # Restrictions on mass based on isochrone
        mass_inf = np.min(pnts[:,0])
        mass_sup = np.max(pnts[:,0])
        
        # Getting a mass to go with the chosen metallicity
        mass = sampled_IMF[i]

        # If metallicity and mass can be used to interpolate from isochrone
        if metal_inf <= MHval <= metal_sup and mass_inf <= mass <= mass_sup:
            est_mag = spl(mass)
            sampled_mags.append(est_mag)

        # Plotting a sample slice
        if i==50: 
            for row in pnts:
                plt.scatter(row[0], row[1], color=colour_from_type(row[2]))
            plt.plot(x, spl(x))
            plt.xlabel("Mass")
            plt.ylabel("Magnitude")

    sampled_mags = np.array(sampled_mags)
    # Building up a 2D array to plot histograms outside the loop
    histplots.append(sampled_mags)




plt.figure()

# Plotting the histograms on top of each other
bins_seq = np.linspace(-10, 10, 100)
for i in histplots:
    plt.hist(i, bins_seq, density=True, histtype='step')
plt.xlabel("Magnitude")
plt.ylabel("Density")

plt.show()
