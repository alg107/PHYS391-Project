import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

def l(ra, dec):
    c = SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg')
    l = c.galactic.l.deg
    return l

lV = np.vectorize(l)

def b(ra, dec):
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    b = c.galactic.b.deg
    return b

#bV = np.vectorize(b)

for tmass in pd.read_table("/users/alex/Data/2mass.tbl",
        delim_whitespace=True,
        usecols=["ra", "dec", "k_m", "j_m"],
        chunksize=10**6):

    for col in tmass.columns:
        print(col)

    tmass = tmass[~((tmass["j_m"]-tmass["k_m"]).between(0.4, 1.0, inclusive=False))]
    print("selected stars")

    print("Done B col")

    tmass = tmass.drop(columns=["j_m"])
    print("Dropped junk cols")

    tmass.to_csv("/users/alex/Data/tmass.min.db", mode='a', index=False)
    print("saved")


print(tmass)
