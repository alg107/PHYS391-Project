import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from lib.Constants import *


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

    plt.ylim(0.0, 1.3) # Adjust as necessary
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
    print("Red Clump Mean:", np.round(popt[1], 3))
    print("Red Clump STDDEV:", np.round(np.abs(popt[2]), 3))
    return popt[1], popt[2]

def pnts_from_samples(fname, bins):
    samples = np.load(fname)

    # Structure of histogram
    bins_seq = np.linspace(xmin, xmax, bins)

    # Tangible histogram with useful return values
    counts,bin_edges = np.histogram(samples, bins_seq)


    # Centres from edges
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    return bin_centres, counts, (xmax-xmin)/BINS, MC_NORM, "MC"

# SECTION FOUR: Method Specific Wrappers

def MC_Points(t):
    return pnts_from_samples(
            "Results/MCSamples/type"+str(t)+"N"+str(N)+".npy",
            BINS
            )

def SALF_Points(t):
    xs = np.load("Results/SALF/xs_t"+str(t)+".npy")
    ys = np.load("Results/SALF/ys_t"+str(t)+".npy")
    return xs, ys, xs[1]-xs[0], SALF_NORM, "SA"
