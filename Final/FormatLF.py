import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps
from scipy.optimize import curve_fit
from lib.Iso import Isochrone
import functools

"""
This script is to be run after samples have been generated
with MonteCarlo.py
"""

## SECTION ONE: Helper Functions & Constant Definition

# Constant definition
typs = [1,2,3]

sigma = 0.05

xmin = -3.5
xmax = 1.0
#NORM = 1.0/1606.98
NORM = 1.0
BINS = 1000 # Kind of arbitrary but not too low

N = int(1e7)

# Gaussian function for fitting to RC
def gauss(x, *p):
    a,b,c,d = p
    y = a*np.exp(-np.power((x-b), 2.0)/(2.0*c**2))+d
    return y

def setup_plot():
    # Some matplotlib initialisation
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    plt.xlabel("$M_{K_s}$")
    plt.ylabel("Luminosity Function (Arbitrary Units)")

    #plt.ylim(0.0, 1.3) # Adjust as necessary
    plt.xlim(xmin, xmax)


colours = {
        1: (203.0/256, 101.0/256, 39.0/256), 
        2: (115.0/256, 114.0/256, 174.0/256),
        3: (219.0/256, 148.0/256, 194.0/256),
        4: (74.0/256, 155.0/256, 122.0/256)
        }

## SECTION TWO: Slightly More Important Functions

def RC_sigma(x,y):
    guess_ps = [1,-1,0.5,0]
    popt, pcov = curve_fit(gauss, x, y, p0=guess_ps)
    print("Red Clump Mean:", popt[1])
    print("Red Clump STDDEV:", np.abs(popt[2]))
    return popt[1], popt[2]

def pnts_from_samples(fname, bins):
    samples = np.load(fname)

    # Structure of histogram
    bins_seq = np.linspace(xmin, xmax, bins)

    # Tangible histogram with useful return values
    counts,bin_edges = np.histogram(samples, bins_seq)


    # Centres from edges
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    return bin_centres, counts, (xmax-xmin)/BINS

    
# SECTION THREE: The Reconstruction (Very Important)

# Plots a given branch from its points (does smoothing)
def reconstruct_LF(bin_centres, counts, step, color, t):

    # Convolving with gaussian
    smoothed = gaussian_filter(counts, sigma/step)

    x = np.linspace(xmin, xmax, 10000)
    spl = UnivariateSpline(bin_centres, smoothed, k=3, s=0) # Cubic
    if t == 1:
        # Just snapping off the exponential background
        # very roughly
        pnts = np.column_stack((bin_centres, smoothed))
        pnts = pnts[(pnts[:,0] < -1.89) | (pnts[:,0] > -0.63)]

        spl1 = UnivariateSpline(pnts[:,0], pnts[:,1], s=0)
        spl2 = UnivariateSpline(
                bin_centres, smoothed-spl1(bin_centres), s=0)

        y1 = spl1(x)
        y2 = spl2(x)

        plt.plot(x, y1, color=colours[4])
        plt.plot(x, y2, color=color)
    else:
        # Otherwise we just have the one spline to plot
        y = spl(x)
        plt.plot(x, y, color=color)

    # Fit parameters to RC
    if t==2:
        RC_sigma(x, spl(x))

    return spl, smoothed

# SECTION FOUR: Method Specific Wrappers

def MC_Points(t):
    return pnts_from_samples(
            "Results/MCSamples/type"+str(t)+"N"+str(N)+".npy",
            BINS
            )

def SALF_Points(t):
    xs = np.load("Results/SALF/xs_t"+str(t)+".npy")
    ys = np.load("Results/SALF/ys_t"+str(t)+".npy")
    return xs, ys, xs[1]-xs[0]

def plot_Method(pnts_fn):
    plt.figure()
    setup_plot()


    spls = []
    for t in typs:
        
        # Calls the above functions
        bcs, counts, step = pnts_fn(t)
        counts = NORM*counts
        spl, smoothed = reconstruct_LF(bcs, counts, step, colours[t], t)
        spls.append(spl)

    x = np.linspace(xmin, xmax, 10000)
    y = x*0
    for spl in spls:
        y = y+spl(x)
    plt.plot(x,y, color="black", linestyle="-.")
    area = simps(y,x)
    print("Area:", area)

    plt.legend([
        "Red Giant Branch",
        "Red Giant Branch Bump",
        "Red Clump",
        "Asymptotic Giant Branch",
        "Total"
        ])

if __name__ == "__main__":
    plot_Method(SALF_Points)
    plt.title("SALF")
    plot_Method(MC_Points)
    plt.title("Monte Carlo")
    # iso = Isochrone()
    # iso.plot()
    plt.show()

