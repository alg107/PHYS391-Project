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


# Chabrier IMF PDF
def chabrier(m):
    if m <= 0:
        return 0
    else:
        return (0.158/(np.log(10)*m))*np.exp(-((np.log10(m)-np.log10(0.079))**2/(2*0.69**2)))

# Normal distribution
def norm(x, m, s):
    return (1/np.sqrt(2*np.pi*s**2))*np.exp(-((x-m)**2/(2*s**2)))

# Mass distr. fn.
def MDF(z):
    return norm(z, 0.0, 0.4)

# jiggles points very very slightly just to get around the restriction on
# not having points with equal x-values
def jiggle_pnts(pnts):
    return np.array([np.random.random()*0.00000001+i for i in pnts])

# The same but for one point
def jiggle_pnt(pnt):
    return np.random.random()*0.00000001+pnt 
jiggle_pntV = np.vectorize(jiggle_pnt)


# Taking the useful stuff from the isochrone table
iso_table = np.loadtxt("iso.db")
iso_table2 = pd.read_csv("isoF5.db")
MH = iso_table[:,1]
masses = iso_table[:,3]
Kmag = iso_table[:,32]
types = iso_table[:,9]
df_arr = np.column_stack((MH, masses, Kmag, classify_stageV(types)))
df = pd.DataFrame(df_arr, columns=["MH", "masses", "Kmag", "types"])

# Plotting these 3 vars in a box just to get a feel for the data
fig = plt.figure()
ax = Axes3D(fig)
plt.title("Isochrone mass-magnitude")
#for typ in [1,2,3]:
#    filt = df[df["types"]==typ]
#    ax.scatter(filt["masses"], filt["Kmag"], filt["MH"], marker=".",
ax.scatter(df["masses"], df["Kmag"], df["MH"], marker=".")
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
#data = np.column_stack((jiggle_pnts(Kmag), masses, [int(classify_stage(x)) for x in types], MH))
data = iso_table2.to_numpy()
data[:,2] = jiggle_pntV(data[:,2])

def sort_pnts(pnts):
    return np.array(sorted(pnts, key=lambda x: x[2]))


def thetainv(M, z, show_plt):
    nearestz = find_nearest(data[:,0], z)
    # Take slice corresponding to closest metallicity and chosen type

    ms = []
    derivs = []

    for typ in [1,2,3]:
        pnts = data[data[:,0]==nearestz]
        pnts = pnts[pnts[:,3]==typ]
        pnts = sort_pnts(pnts)
        # This part deals with all the annoying problems with 
        # creating a spline. i.e. implementing all the manual work
        # I had to do
        if typ == 2:
            # Get all the twists and turns
            maxval = pnts[:,1][np.argmax(pnts[:,2])]
            sec1 = pnts[pnts[:,1]>=maxval]
            sec23 = pnts[pnts[:,1]<maxval]
            sec2 = sec3 = []
            if len(sec23) != 0:
                minval = sec23[:,1][np.argmin(sec23[:,0])]
                sec2 = sec23[sec23[:,1]>=minval]
                sec3 = sec23[sec23[:,1]<minval]
            pnts = [sec1,sec2,sec3]
            pnts = [i for i in pnts if len(i)>2]
        elif typ == 3:
            pnts = pnts[pnts[:,2] < -2.5]
            if True in pnts[:,4]:
                leftpnts = pnts[pnts[:,4]==True]
                leftmin = leftpnts[:,2][np.argmin(leftpnts[:,2])]
                pnts13 = pnts[pnts[:,8]==False]
                sec1 = pnts13[(pnts13[:,2] < leftmin) | (pnts13[:,4]==True)]
                sec2 = pnts13[(pnts13[:,2] > leftmin) & (pnts13[:,4]==False)]
                sec3 = pnts[pnts[:,7]==True]
                pnts = [sec1, sec2, sec3]
            else:
                pnts = [pnts]
            pnts = [i for i in pnts if len(i)>2]
        else:
            if True in pnts[:,4]:
                leftpnts = pnts[pnts[:,4]==True]
                leftmin = leftpnts[:,2][np.argmin(leftpnts[:,2])]
                pnts13 = pnts[pnts[:,6]==False]
                sec1 = pnts13[(pnts13[:,2] < leftmin) | (pnts13[:,4]==True)]
                sec2 = pnts13[(pnts13[:,2] > leftmin) & (pnts13[:,4]==False)]
                sec3 = pnts[pnts[:,5]==True]
                pnts = [sec1, sec2, sec3]
            else:
                pnts = [pnts]
            pnts = [i for i in pnts if len(i)>2]

        #pnts = delete_duplicates(pnts)
        #pnts = np.array(sorted(pnts, key=lambda x: x[0]))
        #print(pnts[:,0])

        # Using a linear spline to connect fixed points
        for sec in pnts:
            #print(len(sec))
            #x = np.linspace(np.min(sec[:,0]),np.max(sec[:,0]),100000)
            if len(sec)<=2: 
                print("0")
                continue
            spl = InterpolatedUnivariateSpline(sec[:,2], sec[:,1], k=1) # s=0
            dv = spl.derivative()
            if show_plt:
                x = np.linspace(np.min(sec[:,2]),np.max(sec[:,2]),100000)
                for row in sec:
                    plt.scatter(row[2], row[1], color=colour_from_type(row[3]))
                plt.plot(x, spl(x))
                #plt.xlim(0.95, 0.958)
                plt.xlabel("Magnitude")
                plt.ylabel("Mass")

            if M < np.min(sec[:,2]) or M > np.max(sec[:,2]):
                continue

            ms.append(spl(M))
            derivs.append(dv(M))
    return ms, derivs

# Doing a few test plots
plt.figure()
test_zs = np.linspace(np.min(data[:,0]), np.max(data[:,0]), 5)
for test_z in test_zs:
    plt.figure()
    thetainv(1, 1, True)
    thetainv(1, 2, True)
    thetainv(1, 3, True)


def phi(M, z):
    ms, derivs = thetainv(M, z, False)
    phi_c = 0
    for i, m in enumerate(ms):
        if m < 0: m=0
        toadd = chabrier(m)*np.abs(derivs[i])
        if np.isnan(toadd): toadd = 0
        #if i == 1: print(toadd)
        phi_c += toadd
    return phi_c

RES = 1000
x = np.linspace(-3.5, 1.0, RES)

#plt.figure()
#for i in [1,2,3]:
#    plt.plot(x, phi(x, -1.0, i))

def Phi(M):
    zs = np.unique(data[:,0])
    phis = np.array([phi(M, z) for z in zs])
    MDFs = np.array([MDF(z) for z in zs])
    ys = phis*MDFs
    I = simps(ys, zs)
    #print(I)
    return I

plt.figure()
total = x*0


# Smoothing and plotting
ys = np.array([Phi(M) for M in x])
#if i!=2: ys *=0.5
sigma = 20 
gaussian = np.exp(-(x/sigma)**2/2)
kernel = Gaussian1DKernel(stddev=sigma)
#smoothed = np.convolve(ys, gaussian, mode="same")
smoothed = convolve(ys, kernel, boundary='extend')

plt.plot(x, ys)
plt.plot(x, smoothed)
plt.xlabel("Magnitude")
plt.ylabel("Luminosity Function (Arbitrary Units)")
print("Done Plotting")
#plt.plot(x, smoothed, linestyle='-.', color='black')

# Saving
np.save("Cache/3xs", x)
np.save("Cache/3ys", ys)
# np.save("Cache/xs", x)
# np.save("Cache/ys1", ysT[0])
# np.save("Cache/ys2", ysT[1])
# np.save("Cache/ys3", ysT[2])
# np.save("Cache/smoothed1", smoothedT[0])
# np.save("Cache/smoothed2", smoothedT[1])
# np.save("Cache/smoothed3", smoothedT[2])
# print("Saved arrays")

plt.show()
