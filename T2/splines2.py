import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, splev
from scipy.integrate import simps
from scipy.ndimage.filters import gaussian_filter1d
from astropy.convolution import convolve, Gaussian1DKernel
from progressbar import progressbar as pb

# Classifies stage based on statement in paper 1
def classify_stage(val):
    # 1: Red Giant
    # 2: RC
    # 3: Asymptotic Giant
    if val<=3:
        return 1
    elif val <= 6:
        return 2
    else:
        return 3 
classify_stageV = np.vectorize(classify_stage)

# Gets a colour given a number from 1-3
def colour_from_type(typ):
    if typ==1:
        return "red"
    elif typ==2:
        return "blue"
    elif typ==3:
        return "green"
    else:
        return "yellow"

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return nodes[np.argmin(dist_2)]

def onclick(event):
    print("Click!")
    global ix, iy
    ix, iy = event.xdata, event.ydata

    # print 'x = %d, y = %d'%(
    #     ix, iy)

    # assign global variable to access outside of function
    global coords
    coords.append((ix, iy))

    return



# Taking the useful stuff from the isochrone table
iso_table = pd.read_csv("isoF4.db")
iso_table['conG'] = False
df = iso_table.copy()

# Plotting these 3 vars in a box just to get a feel for the data
# fig = plt.figure()
# ax = Axes3D(fig)
# plt.title("Isochrone mass-magnitude")
##for typ in [1,2,3]:
##    filt = df[df["types"]==typ]
##    ax.scatter(filt["masses"], filt["Kmag"], filt["MH"], marker=".",
# ax.scatter(df["masses"], df["Kmag"], df["MH"], marker=".")
# ax.set_xlabel("mass")
# ax.set_ylabel("mag")
# ax.set_zlabel("metallicity")

metall = np.sort(df["MH"].unique())

for mh in pb(metall, redirect_stdout=True):
    fig = plt.figure()
    plt.xlim((-3.5, 0.0))
    coords = []
    cid = fig.canvas.mpl_connect('key_press_event', onclick)
    sel = df[df['MH']==mh]
    for i, row in sel.iterrows():
        plt.scatter(row.Kmag, row.masses, color=colour_from_type(row.type))
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    for coord in coords:
        Kmag_c, masses_c = closest_node(coord, [(s.Kmag, s.masses) for _, s in sel.iterrows()])
        df.loc[(df['Kmag']==Kmag_c) & (df['masses']==masses_c), 'conG'] = True

df.to_csv('isoF5.db', index=False)



    
