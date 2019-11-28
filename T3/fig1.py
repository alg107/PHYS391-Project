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

### Section ONE: Middle


