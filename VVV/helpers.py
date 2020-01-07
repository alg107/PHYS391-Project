import numpy as np 

def load_saved_maps():
    return (np.load("Cache/middle_map.npy"),
        np.load("Cache/right_map.npy"),
        np.load("Cache/left_map.npy"),
        np.load("Cache/L_bins.npy", allow_pickle=True),
        np.load("Cache/B_bins.npy", allow_pickle=True)
    )

def nan_to_zero(x):
    if np.isnan(x):
        return 0
    else:
        return x

nan_to_zeroV = np.vectorize(nan_to_zero)

def safe_divide(a, b):
    if b == 0 or np.isnan(b):
        return 0
    elif np.isnan(a):
        return np.nan
    else:
        return a/b

safe_divideV = np.vectorize(safe_divide, otypes=[float])
