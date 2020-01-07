import numpy as np 
import pandas as pd
from astropy.table import Table


tmass= pd.read_csv("/users/alex/Data/tmassF.min.db")
print("imported 2mass")

tmass = tmass.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
print("fixed 2mass")

vvv = pd.read_csv("/users/alex/Data/vvvEQUF.min.db")
print("imported vvv")

vvv = vvv.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
print("fixed vvv")

tmassA = Table.from_pandas(tmass)
vvvA = Table.from_pandas(vvv)

tmassA.write("/users/alex/Data/tmass.fits", format='fits')
vvvA.write("/users/alex/Data/vvv.fits", format='fits')


