import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, Rbf
from scipy.integrate import simps
from scipy.ndimage.filters import gaussian_filter1d
from astropy.convolution import convolve, Gaussian1DKernel
from progressbar import progressbar as pb

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
classify_stageV = np.vectorize(classify_stage)

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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# Chabrier IMF PDF
def chabrier(m):
    if m <= 0:
        return 0
    else:
        return (0.158/(np.log(10)*m))*np.exp(-((np.log10(m)-np.log10(0.079))**2/(2*0.69**2)))

# Normal distribution
def norm(x, m, s):
    return (1/np.sqrt(2*np.pi*s**2))*np.exp(-((x-m)**2/(2*s**2)))

# Mass distr. fn.
def MDF(z):
    return norm(z, 0.0, 0.4)

# jiggles points very very slightly just to get around the restriction on
# not having points with equal x-values
def jiggle_pnts(pnts):
    return np.array([np.random.random()*0.00000001+i for i in pnts])

# The same but for one point
def jiggle_pnt(pnt):
    return np.random.random()*0.00000001+pnt 
jiggle_pntV = np.vectorize(jiggle_pnt)


# Taking the useful stuff from the isochrone table
iso_table = np.loadtxt("iso.db")
iso_table2 = pd.read_csv("isoF5.db")
MH = iso_table[:,1]
masses = iso_table[:,3]
Kmag = iso_table[:,32]
types = iso_table[:,9]
df_arr = np.column_stack((MH, masses, Kmag, classify_stageV(types)))
df = pd.DataFrame(df_arr, columns=["MH", "masses", "Kmag", "types"])
df = df[df["Kmag"].between(-3.5, 1.0)]
# df = df[df["MH"].between(-2.279, 0.198)]
df = df[df["MH"].between(-0.05, 0.0)]



# Plotting these 3 vars in a box just to get a feel for the data
fig = plt.figure()
ax = Axes3D(fig)
plt.title("Isochrone Plot")
for typ in [1]:
    filt = df[df["types"]==typ]
    print(len(filt["Kmag"]))

    x_grid = np.linspace(-3.5, 1.0, 1000*len(filt["Kmag"]))
    y_grid = np.linspace(0, 132651, 1000*len(filt["MH"]))
    B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
    Z = np.zeros((filt["Kmag"].size, filt["masses"].size))

    spline = Rbf(filt["Kmag"],filt["MH"],filt["masses"],
            function='linear')

    Z = spline(B1,B2)
    ax.plot_surface(B1, B2, Z,alpha=0.2)
    ax.scatter(filt["Kmag"],
               filt["MH"],
               filt["masses"],
               marker=".", color=colour_from_type(typ))

ax.set_xlabel("Absolute Magnitude ($M_{K_s}$)")
ax.set_ylabel("Metallicity ($z$)")
ax.set_zlabel("Mass ($m$)")

print("3D isochrone plot done")
plt.show()
