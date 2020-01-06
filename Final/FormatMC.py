import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps
import functools

"""
This script is to be run after samples have been generated
with MonteCarlo.py
"""


# Constant definition
sigma = 0.05

xmin = -3.5
xmax = 1.0
NORM = 1.0/8081.533
BINS = 1000 # Kind of arbitrary but not too low

N = int(1e7)

colours = {
        1: (203.0/256, 101.0/256, 39.0/256), 
        2: (115.0/256, 114.0/256, 174.0/256),
        3: (219.0/256, 148.0/256, 194.0/256),
        4: (74.0/256, 155.0/256, 122.0/256)
        }

# Plots a given branch from its samples
def reconstruct_LF(fname, bins, color, t):
    samples = np.load(fname)

    # Structure of histogram
    bins_seq = np.linspace(xmin, xmax, bins)
    step = (xmax-xmin)/bins

    # Tangible histogram with useful return values
    counts,bin_edges = np.histogram(samples, bins_seq)


    # Centres from edges
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.

    # Convolving with gaussian
    smoothed = gaussian_filter(counts, sigma/step)

    x = np.linspace(xmin, xmax, 10000)
    spl = UnivariateSpline(bin_centres, smoothed) # Cubic
    if t == 1:
        # Just snapping off the exponential background
        # very roughly
        pnts = np.column_stack((bin_centres, smoothed))
        pnts = pnts[(pnts[:,0]<-1.89) | (pnts[:,0]>-0.63)]

        spl1 = UnivariateSpline(pnts[:,0], pnts[:,1])
        spl2 = UnivariateSpline(
                bin_centres, smoothed-spl1(bin_centres))

        y1 = NORM*spl1(x)
        y2 = NORM*spl2(x)

        plt.plot(x, y1, color=colours[4])
        plt.plot(x, y2, color=color)
    else:
        # Otherwise we just have the one spline to plot
        y = NORM*spl(x)
        plt.plot(x, y, color=color)

    return bin_centres, counts, spl, smoothed

typs = [1,2,3]

if __name__ == "__main__":

    plt.figure()

    # Some matplotlib initialisation
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    plt.xlabel("$M_{K_s}$")
    plt.ylabel("Luminosity Function (Arbitrary Units)")

    plt.ylim(0.0, 1.3) # Adjust as necessary
    plt.xlim(xmin, xmax)

    spls = []
    for t in typs:
        
        # Calls the above function
        bcs, counts, spl, smoothed = reconstruct_LF(
                "Results/MCSamples/type"+str(t)+"N"+str(N)+".npy"
                     , BINS, colours[t], t)
        spls.append(spl)

    x = np.linspace(xmin, xmax, 10000)
    y = x*0
    for spl in spls:
        y = y+spl(x)
    plt.plot(x,NORM*y, color="black", linestyle="-.")
    area = simps(NORM*y,x)
    print("Area:", area)

    plt.legend([
        "Red Giant Branch",
        "Red Giant Branch Bump",
        "Red Clump",
        "Asymptotic Giant Branch",
        "Total"
        ])

    plt.show()

