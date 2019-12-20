import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import fitting
from scipy.stats import norm

"""
This is the main file from which several other files are called
"""


# Reconstructs a histogram from data files collected with lf.py
def reconstruct_LF(fname, bins):
    # Loads the samples obviously
    samples = np.load(fname)
    #samples = np.concatenate((samples, np.load("Samples/Magnitudes/sampledmagsN4000000.npy")))

    # Some matplotlib initialisation
    plt.figure()
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # Plotting the LF histogram

    final_bins = bins

    # Structure of histogram
    bins_seq = np.linspace(-3.5, 1.0, final_bins)

    # Plot of histogram
    plt.hist(samples, bins_seq, density=True, histtype='step')

    # Tangible histogram with useful return values
    counts,bin_edges = np.histogram(samples, bins_seq, density=True)

    # Plot limits
    plt.ylim(0.0, 1.0)
    plt.xlim(-3.5, 1.0)

    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.

    # When you get to uncertainties put them here
    #err = np.random.rand(bin_centres.size)*100

    plt.xlabel("$M_{K_s}$")
    plt.ylabel("Luminosity Function (Arbitrary Units)")

    plt.figure()

    plt.xlabel("$M_{K_s}$")
    plt.ylabel("Luminosity Function (Arbitrary Units)")



    return bin_centres, counts



if __name__ == "__main__":
    # Calls the above function
    #bcs, counts = reconstruct_LF("Samples/Magnitudes/sampledmagsN100000000.npy", 150)
    bcs, counts = reconstruct_LF("sampledmagMaster.npy", 200)

    print("Gaussian Fit\n")

    # Plot the histogram bins as points on a scatter plot
    plt.scatter(bcs, counts, marker="+", color="black", s=20, alpha=0.5)

    # Calls the fit_data routine in fitting.py which is quite involved
    # This one is for fitting gaussians
    fitting.fit_data(bcs, counts)

    plt.figure()
    plt.xlabel("$M_{K_s}$")
    plt.ylabel("Luminosity Function (Arbitrary Units)")

    # The same thing again for the skew fit
    plt.scatter(bcs, counts, marker="+", color="black", s=20, alpha=0.5)

    print("\nSkew Fit\n")
    fitting.fit_data_skew(bcs, counts)
    print("Disregard uncertainties for now.")
    plt.show()
