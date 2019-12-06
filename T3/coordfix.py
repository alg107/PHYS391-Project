import numpy as np
import pandas as pd

def fix(x):
    if isinstance(x, str):
        return np.nan
    if x > 180:
        return x-360
    else:
        return x

fixV = np.vectorize(fix, otypes=["float"])
isinstanceV = np.vectorize(isinstance)

for tmass in pd.read_csv("/users/alex/Data/tmass.min.db", chunksize=10**6):

    tmass = tmass[~isinstanceV(tmass["L"], str)]
    tmass = tmass[~isinstanceV(tmass["B"], str)]
    tmass = tmass[~isinstanceV(tmass["k_m"], str)]

    print(tmass["L"])
    tmass["L"] = fixV(tmass["L"])
    tmass["B"] = fixV(tmass["B"])

    tmass.to_csv("/users/alex/Data/tmassfix.min.db", mode='a', index=False)
