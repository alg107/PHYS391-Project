import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d, UnivariateSpline, NearestNDInterpolator
from scipy.integrate import simps
from scipy.ndimage.filters import gaussian_filter1d
from astropy.convolution import convolve, Gaussian1DKernel
from progressbar import progressbar as pb
from scipy.stats import binned_statistic_2d

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
MH = iso_table[:,1]
masses = iso_table[:,3]
Kmag = iso_table[:,32]
types = iso_table[:,9]
df_arr = np.column_stack((MH, masses, Kmag, classify_stageV(types)))
df = pd.DataFrame(df_arr, columns=["MH", "masses", "Kmag", "types"])

df_full = df.copy()

#df = df[df['Kmag'].between(-3.5, 1.0)]
df = df[df['Kmag'].between(-5.0, 2.0)]

def isochrone(m, z, d):
    mz = np.column_stack((d['masses'], d['MH']))
    interpolator = NearestNDInterpolator(mz, d['Kmag'])
    return interpolator(m,z)

def plot_isochrone(df):
    # Plotting these 3 vars in a box just to get a feel for the data
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.title("Isochrone mass-magnitude")

    for typ in [1,2,3]:
       filt = df[df["types"]==typ]
       ax.scatter(filt["masses"], filt["MH"], filt["Kmag"], marker=".", 
               color=colour_from_type(typ))
    #ax.scatter(df["masses"], df["Kmag"], df["MH"], marker=".")
    x = 0.9
    y = -0.7
    ax.scatter(x, y, isochrone(x,y, df), marker=".", color="orange")
    ax.set_xlabel("Mass ($m$)")
    ax.set_ylabel("Metallicity ($z$)")
    ax.set_zlabel("Magnitude ($M_{K_s}$)")
    return plt


plot_isochrone(df)
plt.show()

# plot_isochrone(df[df['Kmag']<=-2.0])


print("3D isochrone plot done")

binx = 200
biny = 25

plot_arr = binned_statistic_2d(df['masses'], df['MH'], df['Kmag'], bins=[binx, biny])
plt.figure
plt.imshow(plot_arr.statistic.T, aspect=binx/biny)
plt.colorbar()





def SA_CDF(M_star):
    df_local = df[df['Kmag'] <= M_star]
    return integrate_df(df_local)
SA_CDF_V = np.vectorize(SA_CDF)

def integrate_df(df_local):
    # This function needs to compute the double integral
    # under the given scatter plot
    return 0

def deriv(MKs_s, CDF):
    spl = UnivariateSpline(MKs_s, CDF, k=3, s=0)
    d = spl.derivative()
    # Could also just return the spline 
    # As this would give a smoother fit
    # if for some reason I wanted to reduce
    # noise
    return d(MKs_s)

MKs_s = np.linspace(-3.5, 1.0, 100)
CDF = SA_CDF_V(MKs_s)
SALF = deriv(MKs_s, CDF)
# plt.figure()
# plt.plot(MKs_s, SALF)

plt.show()
