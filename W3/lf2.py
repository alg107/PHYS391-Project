import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d, UnivariateSpline, splev
from scipy.integrate import simps

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

def norm(x, m, s):
    return (1/np.sqrt(2*np.pi*s**2))*np.exp(-((x-m)**2/(2*s**2)))

def MDF(z):
    return norm(z, 0.0, 0.4)

def jiggle_pnts(pnts):
    return np.array([np.random.random()*0.00000001+i for i in pnts])



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


# Cutting the isochrone into constant metallicity slices
data = np.column_stack((jiggle_pnts(Kmag), masses, [int(classify_stage(x)) for x in types], MH))

def sort_pnts(pnts):
    return np.array(sorted(pnts, key=lambda x: x[0]))


def thetainv(M, z, typ, show_plt):
    nearestz = find_nearest(data[:,3], z)
    # Take slice corresponding to closest metallicity and chosen type
    pnts = data[data[:,3]==nearestz]
    pnts = pnts[pnts[:,2]==typ]
    pnts = sort_pnts(pnts)
    if typ == 2:
        # Get all the twists and turns
        pnts = pnts[pnts[:,0]<0.0]
        maxval = pnts[:,1][np.argmax(pnts[:,0])]
        sec1 = pnts[pnts[:,1]>=maxval]
        sec23 = pnts[pnts[:,1]<maxval]
        sec2 = sec3 = []
        if len(sec23) != 0:
            minval = sec23[:,1][np.argmin(sec23[:,0])]
            sec2 = sec23[sec23[:,1]>=minval]
            sec3 = sec23[sec23[:,1]<minval]
        pnts = [sec1,sec2, sec3]
        pnts = [i for i in pnts if len(i)!=0]
    else:
        #pnts = [pnts]
        pnts = [pnts[pnts[:,0]<0.0]]

    #pnts = delete_duplicates(pnts)
    #pnts = np.array(sorted(pnts, key=lambda x: x[0]))
    #print(pnts[:,0])

    # Using a linear spline to connect fixed points
    ms = []
    derivs = []
    for sec in pnts:
        #x = np.linspace(np.min(sec[:,0]),np.max(sec[:,0]),100000)
        if len(sec)==0: 
            print("0")
            continue
        spl = UnivariateSpline(sec[:,0], sec[:,1], s=0, k=1)
        dv = spl.derivative()
        if show_plt:
            x = np.linspace(np.min(sec[:,0]),np.max(sec[:,0]),100000)
            for row in sec:
                plt.scatter(row[0], row[1], color=colour_from_type(row[2]))
            plt.plot(x, spl(x))
            #plt.xlim(0.95, 0.958)
            plt.xlabel("Magnitude")
            plt.ylabel("Mass")

        ms.append(spl(M))
        derivs.append(dv(M))
    return ms, derivs

plt.figure()
test_zs = np.linspace(np.min(data[:,3]), np.max(data[:,3]), 5)
for test_z in test_zs:
    plt.figure()
    thetainv(1, test_z, 1, True)
    thetainv(1, test_z, 2, True)
    thetainv(1, test_z, 3, True)


def phi(M, z, typ):
    ms, derivs = thetainv(M, z, typ, False)
    phi_c = 0
    for i, m in enumerate(ms):
        if m < 0: m=0
        toadd = chabrier(m)*np.abs(derivs[i])
        if np.isnan(toadd): toadd = 0
        #if i == 1: print(toadd)
        phi_c += toadd
    return phi_c

x = np.linspace(-3.5, 1.0, 1000)

#plt.figure()
#for i in [1,2,3]:
#    plt.plot(x, phi(x, -1.0, i))

def Phi(M, typ):
    zs = np.unique(data[:,3])
    phis = np.array([phi(M, z, typ) for z in zs])
    MDFs = np.array([MDF(z) for z in zs])
    ys = phis*MDFs
    I = simps(ys, zs)
    #print(I)
    return I

plt.figure()
for i in [1,2,3]:
    ys = [Phi(M, i) for M in x]
    sigma = 0.05
    gaussian = np.exp(-(x/sigma)**2/2)
    smoothed = np.convolve(ys, gaussian, mode="same")
    #print(len(ys), len(gaussian), len(smoothed))
    #print(ys)
    plt.plot(x, ys)
    #plt.plot(x, smoothed)

plt.show()

