import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from display import *
from helpers import *


"""
This script recreates fig. 1 of Paper 1 starting with the middle and right images
and then the left image.
"""

rc('text', usetex=True)

# 10**6 chunks means 3 cycles for testvvv2.db and 101 cycles for vvv.db
CHUNKS = 10**6
L = 100
B = 75

## Section ONE: Preparing VVV data
## Involves cutting some bits out and then binning data in this form:
## (K, l, b): (80, 100, 75), 11 < K < 15, -10 < l < 10, -10 < b < 5



def crunch_steps(vvv):

    # Select 11 < K_s < 15
    #vvv = vvv[vvv["KCOR"].between(11.0,15.0)] # For left figure
    vvv = vvv[vvv["KCOR"].between(12.975,13.025)] # For right figure
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
    middle_map = np.zeros((L, B))
    right_map = np.zeros((L, B))
    middle_count = np.zeros((L, B))
    right_count = np.zeros((L, B))

    for i,(_, narrowed) in enumerate(vvv.groupby('L_bin')):
        for j, (_, binned_vals) in enumerate(narrowed.groupby('B_bin')):
            bin_mean = 0
            bin_total = 0
            bin_mean2 = 0
            bin_total2 = 0
            if not binned_vals.empty:
                EJKs = binned_vals['EJK']
                KCOMBERRs = binned_vals['KCOMBERR']
                EJKs = EJKs[~np.isnan(EJKs)]
                KCOMBERRs = KCOMBERRs[~np.isnan(KCOMBERRs)]
                if not EJKs.empty:
                    bin_mean = EJKs.mean()
                    bin_total = len(EJKs)
                if not KCOMBERRs.empty:
                    bin_mean2 = KCOMBERRs.mean()
                    bin_total2 = len(KCOMBERRs)
            middle_map[i,j] = bin_mean
            middle_count[i,j] = bin_total
            right_map[i,j] = bin_mean2
            right_count[i,j] = bin_total2

    print("Built plot arrays")


    #middle_map = nan_to_zeroV(middle_map)
    #right_map = nan_to_zeroV(right_map)

    return middle_map, right_map, middle_count, right_count, L_bins, B_bins

def crunch_middle_right(fname):
    # For no-chunk processing
    vvv = pd.read_csv(fname)
    print("read data")
    middle_map, right_map, middle_count, right_count, L_bins, B_bins = crunch_steps(vvv)
    np.save("middle_map", middle_map)
    np.save("right_map", right_map)
    np.save("L_bins", L_bins)
    np.save("B_bins", B_bins)
    return middle_map, right_map, L_bins, B_bins



def process_chunks(fname):
    # Instantiate the master plot arrays
    middle_map = np.zeros((L, B))
    right_map = np.zeros((L, B))
    middle_count = np.zeros((L, B))
    right_count = np.zeros((L, B))
    L_bins = np.array([])
    B_bins = np.array([])

    counter = 1
    
    for chunk in pd.read_csv(fname, chunksize=CHUNKS):

        # Generate the plot arrays for a chunk
        res = crunch_steps(chunk)
        middle_mapT, right_mapT, middle_countT, right_countT, L_binsT, B_binsT = res

        # Combine these with the previous chunks using a weighted average
        middle_map = safe_divideV(((middle_count*middle_map)
            + (middle_countT*middle_mapT)), 
            ((middle_count+middle_countT)))
        right_map = safe_divideV(((right_count*right_map) 
            + (right_countT*right_mapT)),
            ((right_count+right_countT)))

        # Update the total counts
        middle_count = middle_count+middle_countT
        right_count = right_count+right_countT

        L_bins = L_binsT
        B_bins = B_binsT

        # A counter to keep track of progress
        print(counter)
        counter+=1


    # Save the generated arrays
    np.save("middle_map", middle_map)
    np.save("right_map", right_map)
    np.save("L_bins", L_bins)
    np.save("B_bins", B_bins)
    return middle_map, right_map, L_bins, B_bins

if __name__=="__main__":
    #middle_map, right_map, L_bins, B_bins = crunch_middle_right("testvvv2.db")
    #middle_map, right_map, L_bins, B_bins = process_chunks("/users/alex/Data/vvv.db")
    middle_map, right_map, L_bins, B_bins = process_chunks("testvvv2.db")
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

