import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from display import *
from helpers import *
from matplotlib import colors


"""
This script recreates fig. 1 of Paper 1 starting with the middle and right images
and then the left image.
"""

rc('text', usetex=True)

# 10**6 chunks means 3 cycles for testvvv2.db and 101 cycles for vvv.db
CHUNKS = 10**6
L = 14
B = 14

## Section ONE: Preparing VVV data
## Involves cutting some bits out and then binning data in this form:
## (K, l, b): (80, 100, 75), 11 < K < 15, -10 < l < 10, -10 < b < 5



def crunch_steps(vvv):

    # Select 11 < K_s < 15
    #vvv = vvv[vvv["KCOR"].between(11.0,15.0)] # For left figure

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
    # Uncomment below line later
    #vvv = vvv[~((vvv["JCOR"]-vvv["KCOR"]).between(0.4, 1.0, inclusive=False))]
    print("Selected mainly Red Giant stars")

    # Next use pd.cut to bin data

    vvv['L_bin'], L_bins = pd.cut(vvv['L'],
            pd.interval_range(start=-10.0, end=10.0, periods=L), retbins=True)
    print("Cut L")
    vvv['B_bin'], B_bins = pd.cut(vvv['B'],
            pd.interval_range(start=-10.0, end=5.0, periods=B), retbins=True)
    print("Cut B")

    # Gets the different bin values and sorts them
    # to be iterated over
    #K_bins = vvv.K_bin.unique()

    ### Section TWO: Middle & Right

    # Creates the array where the displayed values will
    # be stored
    left_map = np.zeros((L, B))

    for i,(b1, narrowed) in enumerate(vvv.groupby('L_bin')):
        print("B1:", b1)
        for j, (b2, binned_vals) in enumerate(narrowed.groupby('B_bin')):
            print("B2:", b2)
            bin_median = 0
            if not binned_vals.empty:
                KDELs = binned_vals['KDEL']
                if not KDELs.empty:
                    bin_median = KDELs.median()
            left_map[i,j] = bin_median

    print("Built plot arrays")

    return left_map, L_bins, B_bins

def crunch_middle_right(fname):
    # For no-chunk processing
    vvv = pd.read_csv(fname, usecols=["KDEL", "L", "B"])
    print("read data")
    left_map, L_bins, B_bins = crunch_steps(vvv)
    np.save("left_map", left_map)
    np.save("L_bins", L_bins)
    np.save("B_bins", B_bins)
    return left_map, L_bins, B_bins




if __name__=="__main__":
    #middle_map, right_map, L_bins, B_bins = crunch_middle_right("testvvv2.db")
    #middle_map, right_map, L_bins, B_bins = process_chunks("/users/alex/Data/vvv.db")
    left_map, L_bins, B_bins = crunch_middle_right("/users/alex/Data/cross.csv")
    present_left(left_map, L_bins, B_bins)

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
# [Not done]

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

