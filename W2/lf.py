import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

# 1: Constant Definition

bulge_age = 10 #Gyr
N = int(1e3)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

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

sampled_metallicities = np.random.normal(metallicity_mean, metallicity_std, N)

plt.figure()
plt.title("Metallicity Distribution")
metallicity_bins_seq = np.linspace(-2.279, 0.198, 39)
plt.hist(sampled_metallicities, metallicity_bins_seq, density=True, histtype='step')

# 2.2: IMF

# Does it matter that this has natural log and the log in Chabrier is unspecified?
sampled_IMF = np.random.lognormal(IMF_mean, IMF_std, N)

plt.figure()
plt.title("IMF Distribution")
plt.hist(sampled_IMF, IMF_bins, density=True, histtype='step')

# 3: Isochrones

MM = np.column_stack((sampled_metallicities, sampled_IMF))

iso_table = np.loadtxt("iso.db")
MH = iso_table[:,1]
masses = iso_table[:,5]
Kmag = iso_table[:,32]
types = iso_table[:,9]

fig = plt.figure()
ax = Axes3D(fig)
plt.title("Isochrone mass-magnitude")
ax.scatter(masses, Kmag, MH, marker=".")
ax.set_xlabel("mass")
ax.set_ylabel("mag")
ax.set_zlabel("metallicity")


# First attempt at generating a LF
metal_inf = np.min(MH)
metal_sup = np.max(MH)
print(metal_inf, metal_sup)
histplots = []

plt.figure()
for type_ in [1,2,3]:
    sampled_mags = []
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
        data = np.column_stack((mass_i, mags_i, types_i))
        redgiants = data[data[:,2] == 1] 
        RC = data[data[:,2] == 2] 
        asymp = data[data[:,2] == 3] 

        pnts = data[data[:,2] == type_] 


        x = np.linspace(np.min(pnts[:,0]),np.max(pnts[:,0]),1000)
        spl = interp1d(pnts[:,0], pnts[:,1])

        mass_inf = np.min(pnts[:,0])
        mass_sup = np.max(pnts[:,0])
        #print(mass_inf, mass_sup)
        
        mass = sampled_IMF[i]
        #print(MHval)
        if metal_inf <= MHval <= metal_sup and mass_inf <= mass <= mass_sup:
            est_mag = spl(mass)
            sampled_mags.append(est_mag)
        if i==50: 
            for row in pnts:
                plt.scatter(row[0], row[1], color=colour_from_type(row[2]))
            plt.plot(x, spl(x))
            plt.xlabel("Mass")
            plt.ylabel("Magnitude")

    sampled_mags = np.array(sampled_mags)
    histplots.append(sampled_mags)




plt.figure()

bins_seq = np.linspace(-10, 10, 100)
for i in histplots:
    plt.hist(i, bins_seq, density=True, histtype='step')
plt.xlabel("Magnitude")
plt.ylabel("Density")

plt.show()
