import numpy as np
import pandas as pd
from astropy.table import Table



# This is for taking the fits file given by Nway and converting it into the form
# Used by left2.py to bin and present

nway = Table.read("/users/alex/Data/cross.fits", format="fits")
useful = nway.to_pandas()
print(useful.columns)
useful["KDEL"] = useful["TMASS_K"]-useful["VVV_K"]
useful.drop(useful.columns.difference(['KDEL','VVV_L', 'VVV_B', "TMASS_K", "VVV_K"]), 1, inplace=True)
useful.rename(columns={"VVV_L": "L", "VVV_B": "B"}, inplace=True)
useful = useful[useful["TMASS_K"]!=-99]
useful.to_csv("/users/alex/Data/cross.csv", index=False)
