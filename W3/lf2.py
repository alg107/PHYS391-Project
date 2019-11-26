import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d, UnivariateSpline, splev

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


def chabrier(m):
    return (0.158/(np.log(10)*m))*np.exp(-((np.log10(m)-np.log10(0.079))**2/(2*0.69**2)))

# 3: Isochrones

#MM = np.column_stack((sampled_metallicities, sampled_IMF))

# Taking the useful stuff from the isochrone table
iso_table = np.loadtxt("iso.db")
MH = iso_table[:,1]
masses = iso_table[:,3]
Kmag = iso_table[:,32]
types = iso_table[:,9]

# Plotting these 3 vars in a box just to get a feel for the data
fig = plt.figure()
ax = Axes3D(fig)
plt.title("Isochrone mass-magnitude")
ax.scatter(masses, Kmag, MH, marker=".")
ax.set_xlabel("mass")
ax.set_ylabel("mag")
ax.set_zlabel("metallicity")

print("3D isochrone plot done")

def delete_duplicates(pnts):
    xs = []
    counter = 0
    for i, row in enumerate(pnts):
        i = i-counter
        if row[0] in xs:
            #print("deleted")
            pnts = np.delete(pnts, i, axis=0)
            counter+= 1 
        else:
            xs.append(row[0])
    return pnts


plt.figure()
# Cutting the isochrone into constant metallicity slices
data = np.column_stack((Kmag, masses, [int(classify_stage(x)) for x in types], MH))

def thetainv(M, z, typ):
    nearestz = find_nearest(data[:,3], z)
    # Take slice corresponding to closest metallicity and chosen type
    pnts = data[data[:,3]==nearestz]
    pnts = pnts[pnts[:,2]==typ]

    #pnts = delete_duplicates(pnts)
    #pnts = np.array(sorted(pnts, key=lambda x: x[0]))
    #print(pnts[:,0])

    #if len(pnts)==0: continue
    # Using a linear spline to connect fixed points
    #x = np.linspace(np.min(pnts[:,0]),np.max(pnts[:,0]),100000)
    #spl = UnivariateSpline(pnts[:,0], pnts[:,1])
    #splder = spl.derivative()
    plt.xlabel("Magnitude")
    plt.ylabel("Mass")
    for row in pnts:
        plt.scatter(row[0], row[1], color=colour_from_type(row[2]))
    #return m, deriv

thetainv(1, 0.8, 1)
thetainv(1, 0.8, 2)
thetainv(1, 0.8, 3)

def phi(M, z, typ):
    # Need to account for multiple solutions maybe
    m, deriv = thetainv(M, z, typ)
    return chabrier(m)*np.abs(deriv)


plt.show()

