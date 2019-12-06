import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from display import *
from helpers import *
from astroML.crossmatch import crossmatch_angular

"""
This script recreates fig. 1 of Paper 1 starting with the middle and right images
and then the left image.
"""

rc('text', usetex=True)

L = 14
B = 14

def subscr(ind, arr):
    return arr[ind]
subscrV = np.vectorize(subscr)


## Section ONE: Preparing VVV data
## Involves cutting some bits out and then binning data in this form:
## (K, l, b): (80, 100, 75), 11 < K < 15, -10 < l < 10, -10 < b < 5

tmass= pd.read_csv("/users/alex/Data/tmassF.min.db")
print("imported 2mass")

tmass = tmass.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
print("fixed 2mass")

vvv = pd.read_csv("/users/alex/Data/vvvEQUF.min.db")
print("imported vvv")

vvv = vvv.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
print("fixed vvv")

X1 = vvv[["RA2000", "DEC2000"]].to_numpy()
X2 = tmass[["ra", "dec"]].to_numpy()
dist, ind = crossmatch_angular(X1, X2, max_distance=(1.0/36000))
print("Done crossmatching")

vvv['DIST'] = dist
vvv["IND"] = ind

vvv = vvv[vvv['DIST']!=np.inf]
print(vvv)

print("Max of gen ind: ", np.max(vvv["IND"]))
print("Length of tmass: ", len(tmass['k_m']))

vvv["K2MASS"] = np.array([tmass['k_m'][i] for i in vvv["IND"]])
print("Subscripted K2MASS vals")

del tmass

vvv["KDEL"] = vvv["K2MASS"]-vvv["K"]
print("Calculated differences")
vvv = vvv.drop(columns=["K2MASS", "RA2000", "DEC2000", "KCOR", "DIST", "IND", "K"])
#vvv = vvv.drop(columns=["RA2000", "DEC2000",  "IND"])
print("Dropped unnecessary cols")
vvv.to_csv("/users/alex/Data/diffs.csv", index=False)
print("Done and saved")

# Bin and plot

