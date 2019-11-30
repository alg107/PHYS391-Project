import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
import dask.dataframe as dd


"""
This script recreates fig. 1 of Paper 1 starting with the middle and right images
and then the left image.
"""

rc('text', usetex=True)

## Section ONE: Preparing VVV data
## Involves cutting some bits out and then binning data in this form:
## (K, l, b): (80, 100, 75), 11 < K < 15, -10 < l < 10, -10 < b < 5

def crunch_middle_right(fname):

    vvv = pd.read_csv(fname)
    print("read data")


    # Select 11 < K_s < 15
    vvv = vvv[vvv["KCOR"].between(11.0,15.0)]
    print("Trimmed K")

    # Select -10 < l < 10
    vvv = vvv[vvv["L"].between(-10.0, 10.0)]
    #vvv = vvv[vvv["L"].between(-0.2, 0.2)]
    print("Trimmed L")

    # Select -10 < b < 5
    vvv = vvv[vvv["B"].between(-10.0, 5.0)]
    #vvv = vvv[vvv["B"].between(-2.7, -2.3)]
    print("Trimmed B")


    # Cut out 0.4 < J - K_s < 1.0
    # This is to select mainly red giant stars
    vvv = vvv[(vvv["JCOR"]-vvv["KCOR"]).between(0.4, 1.0, inclusive=False)]
    print("Selected mainly Red Giant stars")

    # Next use pd.cut to bin data

    vvv['K_bin'] = pd.cut(vvv['KCOR'], np.linspace(11.0, 15.0, 80))
    print("Cut K")
    vvv['L_bin'] = pd.cut(vvv['L'], np.linspace(-10.0, 10.0, 100))
    print("Cut L")
    vvv['B_bin'] = pd.cut(vvv['B'], np.linspace(-10.0, 5.0, 75))
    print("Cut B")

    # Gets the different bin values and sorts them
    # to be iterated over
    #K_bins = vvv.K_bin.unique()
    L_bins = np.sort(vvv.L_bin.unique())
    B_bins = np.sort(vvv.B_bin.unique())

    ### Section TWO: Middle & Right

    # Creates the array where the displayed values will
    # be stored
    middle_map = np.zeros((len(L_bins), len(B_bins)))
    right_map = np.zeros((len(L_bins), len(B_bins)))

    for i, Lbin in enumerate(L_bins):
        for j, Bbin in enumerate(B_bins):
            binned_vals = vvv[vvv['L_bin']==Lbin][vvv['B_bin']==Bbin]
            bin_mean = binned_vals['EJK'].mean()
            middle_map[i,j] = bin_mean
            bin_mean2 = binned_vals['KCOMBERR'].mean()
            right_map[i,j] = bin_mean2

    print("Built plot arrays")

    np.save("middle_map", middle_map)
    np.save("right_map", right_map)
    np.save("L_bins", L_bins)
    np.save("B_bins", B_bins)

    print("Saved plot arrays")

    return middle_map, right_map, L_bins, B_bins


def present_middle_right(middle_map, right_map, L_bins, B_bins):
    # Getting dimensions
    # left, right, bottom, top
    extent = [L_bins[0].left, L_bins[-1].right, B_bins[0].left, B_bins[-1].right]

    # Display this array
    plt.figure()
    plt.imshow(middle_map, cmap='gnuplot', origin='lower', extent=extent)
    plt.colorbar(orientation="horizontal", label="$E(J-K_s)(mag)$")
    plt.contour(middle_map, [0.9], colors="white", extent=extent)
    plt.xlabel("$l(^{\circ})$")
    plt.ylabel("$b(^{\circ})$")
    # Needs axes
    plt.figure()
    plt.imshow(right_map, cmap='gnuplot', origin='lower', extent=extent)
    plt.colorbar(orientation="horizontal", label="$\langle \sigma_{K_s} \\rangle (mag)$")
    plt.contour(right_map, [0.06], colors="white", extent=extent)
    plt.xlabel("$l(^{\circ})$")
    plt.ylabel("$b(^{\circ})$")
    # Needs axes
    print("plotted both plots")


    # Masking for both...

    plt.show()

def load_saved_maps():
    return np.load("middle_map.npy"), np.load("right_map.npy"), np.load("L_bins.npy", allow_pickle=True), np.load("B_bins.npy", allow_pickle=True)

if __name__=="__main__":
    middle_map, right_map, L_bins, B_bins = crunch_middle_right("testvvv2.db")
    #middle_map, right_map, L_bins, B_bins = load_saved_maps()
    present_middle_right(middle_map, right_map, L_bins, B_bins)

### Section FOUR: Left

## Preparing the 2MASS data

# 2mass = pd.read_csv('2mass.db')


# # Select 12 < K_s < 13
# 2mass = 2mass[12.0 < 2mass["KCOR"] < 13.0]

# # Select -10 < l < 10
# 2mass = 2mass[-10.0 < 2mass["L"] < 10.0]

# # Select -10 < b < 5
# 2mass = 2mass[-10.0 < 2mass["B"] < 5.0]


# # Cut out 0.4 < J - K_s < 1.0
# # This is to select mainly red giant stars
# vvv = vvv[(vvv["JCORR"]-vvv["KCOR"]) < 0.4]
# vvv = vvv[(vvv["JCORR"]-vvv["KCOR"]) > 1.0]

# # Next use pd.cut to bin data

# 2mass['K_bin'] = pd.cut(2mass['KCOR'], 80)
# 2mass['L_bin'] = pd.cut(2mass['L'], 100)
# 2mass['B_bin'] = pd.cut(2mass['B'], 75)


# left_map = np.zeros((len(L_bins), len(B_bins)))

# # Needs to be redone for left image
# for i, Lbin in enumerate(L_bins):

    # for j, Bbin in enumerate(B_bins):
        # binned_vals = vvv[vvv['L_bin']==Lbin][vvv['B_bin']==Bbin]
        # bin_mean = binned_vals['KCOMBERR'].mean()
        # right_map[i,j] = bin_mean

