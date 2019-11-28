import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
This script recreates fig. 1 of Paper 1 starting with the middle and right images
and then the left image.
"""

## Section ONE: Preparing VVV data
## Involves cutting some bits out and then binning data in this form:
## (K, l, b): (80, 100, 75), 11 < K < 15, -10 < l < 10, -10 < b < 5

vvv = pd.read_csv('vvv.db')


# Select 11 < K_s < 15
vvv = vvv[11.0 < vvv["KCORR"] < 15.0]

# Select -10 < l < 10
vvv = vvv[-10.0 < vvv["L"] < 10.0]

# Select -10 < b < 5
vvv = vvv[-10.0 < vvv["B"] < 5.0]


# Cut out 0.4 < J - K_s < 1.0
# This is to select mainly red giant stars
vvv = vvv[(vvv["JCORR"]-vvv["KCORR"]) < 0.4]
vvv = vvv[(vvv["JCORR"]-vvv["KCORR"]) > 1.0]

# Next use pd.cut to bin data

vvv['K_bin'] = pd.cut(vvv['KCORR'], 80)
vvv['L_bin'] = pd.cut(vvv['L'], 100)
vvv['B_bin'] = pd.cut(vvv['B'], 75)

# Gets the different bin values and sorts them
# to be iterated over
#K_bins = vvv.K_bin.unique()
L_bins = np.sort(vvv.L_bin.unique())
B_bins = np.sort(vvv.B_bin.unique())

### Section TWO: Middle

# Creates the array where the displayed values will
# be stored
middle_map = np.zeros((len(L_bins), len(B_bins)))

for i, Lbin in enumerate(L_bins):
    for j, Bbin in enumerate(B_bins):
        binned_vals = vvv[vvv['L_bin']==Lbin][vvv['B_bin']==Bbin]
        bin_mean = binned_vals['EJK'].mean()
        middle_map[i,j] = bin_mean

# Display this array
plt.figure()
plt.imshow(middle_map, cmap='gnuplot')
plt.colorbar()
# Needs axes

# Masking...

### Section THREE: Right

# Creates the array where the displayed values will
# be stored
right_map = np.zeros((len(L_bins), len(B_bins)))

for i, Lbin in enumerate(L_bins):
    for j, Bbin in enumerate(B_bins):
        binned_vals = vvv[vvv['L_bin']==Lbin][vvv['B_bin']==Bbin]
        bin_mean = binned_vals['KCOMBERR'].mean()
        right_map[i,j] = bin_mean

# Display this array
plt.figure()
plt.imshow(right_map, cmap='gnuplot')
plt.colorbar()
# Needs axes

# Masking...

### Section FOUR: Left
