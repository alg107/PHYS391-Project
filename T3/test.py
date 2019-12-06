import numpy as np
import pandas as pd


fname = "/users/alex/Data/vvvEQU.db"
CHUNKS = 10**6

counter = 1

for vvv in pd.read_csv(fname, chunksize=CHUNKS):

    # Select 11 < K_s < 15
    vvv = vvv[vvv["K"].between(12.0,13.0)] # For left figure
    #vvv = vvv[vvv["KCOR"].between(12.0,13.0)] # For left figure
    #vvv = vvv[vvv["KCOR"].between(12.975,13.025)] # For right figure
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
    vvv = vvv[~((vvv["JCOR"]-vvv["KCOR"]).between(0.4, 1.0, inclusive=False))]
    print("Selected mainly Red Giant stars")

    vvv = vvv.drop(columns=["JCOR"])
    

    vvv.to_csv("/users/alex/Data/vvvEQU.min.db", mode='a', index=False)

    print(counter)
    counter += 1
