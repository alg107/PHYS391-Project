import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps
from lib.Iso import Isochrone
from lib.Helpers import (gauss, setup_plot, colours,
                        RC_sigma, pnts_from_samples, 
                        MC_Points, SALF_Points)
from lib.Constants import *


"""
This script is to be run after samples have been generated
with MonteCarlo.py and SALF.py

It smooths and presents the data as well as doing some nice
analysis on it
"""

# SECTION THREE: The Reconstruction (Very Important)

# Plots a given branch from its points (does smoothing)
def reconstruct_LF(bin_centres, counts, step, color, t, method):

    # Convolving with gaussian
    smoothed = gaussian_filter(counts, sigma/step)

    x = np.linspace(xmin, xmax, 10000)
    spl = UnivariateSpline(bin_centres, smoothed, k=1, s=0) # Cubic
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


def plot_Method(pnts_fn):
    plt.figure()
    setup_plot()


    spls = []
    for t in typs:
        
        # Calls the above functions
        bcs, counts, step, NORM, method = pnts_fn(t)
        counts = NORM*counts
        spl, smoothed = reconstruct_LF(bcs, counts, step, colours[t], t, method)
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
    print("SALF")
    plot_Method(SALF_Points)
    plt.title("SALF")
    print()
    print("Monte Carlo")
    plot_Method(MC_Points)
    plt.title("Monte Carlo")
    # iso = Isochrone()
    # iso.plot()
    plt.show()

