import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d, UnivariateSpline, NearestNDInterpolator
from progressbar import ProgressBar as pb
from scipy.stats import binned_statistic_2d

types = [1,2,3]

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

def find_neighbour(array, value):
    # Array must be sorted and unique
    # Returns index
    array = np.asarray(array)
    dists = (np.abs(array-value))
    dist = dists.min() 
    idx = dists.argmin()
    first = array[idx]
    if idx+1 == len(array):
        return -1
    elif idx == 0:
        return 0
    elif dists[idx+1] < dists[idx-1]:
        return idx
    else:
        return idx-1


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


class Isochrone():
    #def __init__(self, binx=750, biny=25, fname="iso.db"):
    def __init__(self, binx=750, biny=25, fname="iso_big.db"):
        # Taking the useful stuff from the isochrone table
        iso_table = np.loadtxt(fname)
        MH = iso_table[:,1]
        masses = iso_table[:,3]
        Kmag = iso_table[:,32]
        types = iso_table[:,9]
        df_arr = np.column_stack((MH, masses, Kmag, classify_stageV(types)))
        df = pd.DataFrame(df_arr, columns=["MH", "masses", "Kmag", "types"])
        # df_full = df.copy()

        #df = df[df['Kmag'].between(-3.5, 1.0)]
        df = df[df['Kmag'].between(-5.0, 2.0)]
        df['Kmag'] = jiggle_pntV(df['Kmag'])

        # Insert filtering code for types==3

        self.df = df

        self.df_ret = binned_statistic_2d(df['masses'], df['MH'], df['Kmag'], bins=[binx, biny])
        self.gen_splines()

    def gen_splines(self):
        zs = np.sort(np.unique(self.df.MH))
        self.zs = zs
        spls = {}
        for z in zs:
            spls[z] = []
            df_local = self.df[self.df['MH']==z]
            df_local = df_local.drop_duplicates(subset=['masses'])
            df_local = df_local.sort_values(by="masses")
            for typ in types:
                df2 = df_local[df_local.types==typ]

                mmin = df2.masses.min()
                mmax = df2.masses.max()
                spl = UnivariateSpline(df2.masses, df2.Kmag, k=3, s=0)

                spls[z].append((spl, mmin, mmax))
        self.spl_dict = spls
        return spls


    def plot(self, df=None):

        if df is None:
            df = self.df
        # Plotting these 3 vars in a box just to get a feel for the data
        fig = plt.figure()
        ax = Axes3D(fig)
        plt.title("Isochrone mass-magnitude")

        for typ in types:
           filt = df[df["types"]==typ]
           ax.scatter(filt["masses"], filt["MH"], filt["Kmag"], marker=".", 
                   color=colour_from_type(typ))
        #ax.scatter(df["masses"], df["Kmag"], df["MH"], marker=".")
        x = 0.9
        y = -0.7
        #ax.scatter(x, y, isochrone(x,y, df), marker=".", color="orange")
        ax.set_xlabel("Mass ($m$)")
        ax.set_ylabel("Metallicity ($z$)")
        ax.set_zlabel("Magnitude ($M_{K_s}$)")
        return plt

    def colour_plot(self):
        plt.figure()
        df = self.df
        plot_arr = self.df_ret
        extent = [plot_arr.x_edge[0],
                  plot_arr.x_edge[-1],
                  plot_arr.y_edge[0], plot_arr.y_edge[-1]]
        plt.imshow(plot_arr.statistic.T, aspect='auto',  extent=extent)
        plt.colorbar()

    def interpolate2(self, m, z):
        x_idx = find_neighbour(self.df_ret.x_edge, m)
        y_idx = find_neighbour(np.flip(self.df_ret.y_edge), z)
        return self.df_ret.statistic[x_idx, y_idx]
    interpolate2V = np.vectorize(interpolate2)

    def interpolate(self, m, z):
        closest_z = find_nearest(self.zs, z)
        for i, typ in enumerate(types):
            spl, mmin, mmax = self.spl_dict[closest_z][i]
            if m < mmin or m > mmax: 
                continue 
            else:
                return spl(m)
        return np.nan
    interpolateV = np.vectorize(interpolate)

    def plot_slice(self, z, w_spl=True):
        closest_z = find_nearest(self.zs, z)
        local_df = self.df[self.df["MH"]==closest_z]
        pl = plt.figure()
        for i, typ in enumerate(types):
            df2 = local_df[local_df["types"]==typ]
            plt.scatter(df2["masses"], df2["Kmag"], color=colour_from_type(typ))
            spl, mmin, mmax = self.spl_dict[closest_z][i]
            x = np.linspace(mmin, mmax, 100000)
            y = spl(x)
            plt.plot(x,y)
        return pl




if __name__=="__main__":

    iso = Isochrone()
    iso.plot()

    print("3D isochrone plot done")

    iso.gen_splines()
    val = iso.interpolate(0.871, -0.744)
    print(val)
    metall = np.unique(iso.df['MH'])
    print(metall)

    # iso.colour_plot()
    # print("Colour plot done")
    iso.plot_slice(-2.152)
    iso.plot_slice(-1.644)
    iso.plot_slice(-1.136)
    iso.plot_slice(-0.36)
    iso.plot_slice(0.134)

    plt.show()

