import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc


"""
This script recreates fig. 1 of Paper 1 starting with the middle and right images
and then the left image.
"""

rc('text', usetex=True)
CHUNKS = 10**6
L = 100
B = 75

## Section ONE: Preparing VVV data
## Involves cutting some bits out and then binning data in this form:
## (K, l, b): (80, 100, 75), 11 < K < 15, -10 < l < 10, -10 < b < 5

def nan_to_zero(x):
    if np.isnan(x):
        return 0
    else:
        return x

nan_to_zeroV = np.vectorize(nan_to_zero)


def crunch_steps(vvv):

    # Select 11 < K_s < 15
    #vvv = vvv[vvv["KCOR"].between(11.0,15.0)]
    vvv = vvv[vvv["KCOR"].between(12.975,13.025)]
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

    #vvv['K_bin'], K_bins = pd.cut(vvv['KCOR'], pd.interval_range(start=11.0, end=15.0, periods=80), retbins=True)
    #print("Cut K")
    vvv['L_bin'], L_bins = pd.cut(vvv['L'], pd.interval_range(start=-10.0, end=10.0, periods=L), retbins=True)
    print("Cut L")
    vvv['B_bin'], B_bins = pd.cut(vvv['B'], pd.interval_range(start=-10.0, end=5.0, periods=B), retbins=True)
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
                #binned_vals = vvv[vvv['L_bin']==Lbin][vvv['B_bin']==Bbin]
                bin_mean = binned_vals['EJK'].mean()
                bin_total = binned_vals['EJK'].size
                bin_mean2 = binned_vals['KCOMBERR'].mean()
                bin_total2 = binned_vals['KCOMBERR'].size
            middle_map[i,j] = bin_mean
            middle_count[i,j] = bin_total
            right_map[i,j] = bin_mean2
            right_count[i,j] = bin_total2

    print("Built plot arrays")


    #middle_map = nan_to_zeroV(middle_map)
    #right_map = nan_to_zeroV(right_map)

    return middle_map, right_map, middle_count, right_count, L_bins, B_bins

def crunch_middle_right(fname):
    vvv = pd.read_csv(fname)
    print("read data")
    middle_map, right_map, middle_count, right_count, L_bins, B_bins = crunch_steps(vvv)
    np.save("middle_map", middle_map)
    np.save("right_map", right_map)
    np.save("L_bins", L_bins)
    np.save("B_bins", B_bins)
    return middle_map, right_map, L_bins, B_bins

def present_middle_right(middle_map, right_map, L_bins, B_bins):
    # Getting dimensions
    # left, right, bottom, top
    extent = [L_bins[0].left, L_bins[-1].right, B_bins[0].left, B_bins[-1].right]

    middle_map = np.fliplr(middle_map.T)
    right_map = np.fliplr(right_map.T)

    # Display this array
    plt.figure()
    plt.imshow(middle_map, cmap='gnuplot', origin='lower', extent=extent, vmin=0.0, vmax=3.0)
    plt.colorbar(orientation="horizontal", label="$E(J-K_s)(mag)$")
    plt.contour(middle_map, [0.9], colors="white", extent=extent)
    plt.xlabel("$l(^{\circ})$")
    plt.ylabel("$b(^{\circ})$")
    # Needs axes
    plt.figure()
    plt.imshow(right_map, cmap='gnuplot', origin='lower', extent=extent, vmin=0.0, vmax=0.18)
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

def safe_divide(a, b):
    if b == 0 or np.isnan(b):
        return 0
    elif np.isnan(a):
        return np.nan
    else:
        return a/b

def safe_multiply(a, b):
    if a==0 or b==0:
        return 0
    elif np.isnan(a) or np.isnan(b):
        return np.nan
    else:
        return a*b


safe_divideV = np.vectorize(safe_divide, otypes=[float])
safe_multiplyV = np.vectorize(safe_multiply, otypes=[float])

def process_chunks(fname):
    middle_map = np.zeros((L, B))
    right_map = np.zeros((L, B))
    middle_count = np.zeros((L, B))
    right_count = np.zeros((L, B))
    L_bins = np.array([])
    B_bins = np.array([])

    counter = 1
    
    for chunk in pd.read_csv(fname, chunksize=CHUNKS):

        middle_mapT, right_mapT, middle_countT, right_countT, L_binsT, B_binsT = crunch_steps(chunk)
        middle_map = safe_divideV((safe_multiplyV(middle_count,middle_map) + safe_multiplyV(middle_countT,middle_mapT)), (middle_count+middle_countT))
        right_map = safe_divideV((safe_multiplyV(right_count,right_map) + safe_multiplyV(right_countT,right_mapT)), (right_count+right_countT))
        middle_count = middle_count+middle_countT
        right_count = right_count+right_countT
        #L_bins = np.unique(np.concatenate((L_bins, L_binsT)))
        #B_bins = np.unique(np.concatenate((B_bins, B_binsT)))
        L_bins = L_binsT
        B_bins = B_binsT
        print(counter)
        counter+=1


    np.save("middle_map", middle_map)
    np.save("right_map", right_map)
    np.save("L_bins", L_bins)
    np.save("B_bins", B_bins)
    return middle_map, right_map, L_bins, B_bins

if __name__=="__main__":
    #middle_map, right_map, L_bins, B_bins = crunch_middle_right("testvvv2.db")
    middle_map, right_map, L_bins, B_bins = load_saved_maps()
    #middle_map, right_map, L_bins, B_bins = process_chunks("/users/alex/data/vvv.db")
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

