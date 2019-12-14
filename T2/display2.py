import matplotlib.pyplot as plt
import numpy as np 
from astropy.convolution import convolve, Gaussian1DKernel
import matplotlib as mpl
from scipy.interpolate import UnivariateSpline

mpl.rc('text', usetex=True)

def exp_model(coeff, x):
    A, B = coeff
    return A*np.exp(2*x) + B

plt.figure()

x = np.load("Cache/xs.npy")
ys1 = np.load("Cache/ys1.npy")
ys2 = np.load("Cache/ys2.npy")
ys3 = np.load("Cache/ys3.npy")
ys_full = np.load("Cache/3ys.npy")
# 0.63, -1.89

SCALE = 700/0.1323
ys = [ys1*SCALE, ys2*SCALE, ys3*SCALE, ys_full*SCALE]
total = x*0

total = ys1*0

colours = {
        1: (203.0/256, 101.0/256, 39.0/256), 
        2: (115.0/256, 114.0/256, 174.0/256),
        3: (219.0/256, 148.0/256, 194.0/256),
        4: 'brown',
        5: (74.0/256, 155.0/256, 122.0/256)
        }

for i, y in enumerate(ys):

    sigma = 0
    if i+1!=4:
        sigma = 15
    else:
        sigma = 9

    kernel = Gaussian1DKernel(stddev=sigma)
    smoothed = convolve(y, kernel, boundary='extend')
    if i+1!=4:
        total += smoothed

    if i+1 == 1:
        pnts = np.column_stack((x, smoothed))
        pnts = pnts[(pnts[:,0] < -1.89) | (pnts[:,0] > -0.63)]
        plt.plot(pnts[:,0], pnts[:,1], color=colours[5])
        spl = UnivariateSpline(pnts[:,0], pnts[:,1])
        smoothed = smoothed - spl(x)
        plt.xlim(-3.5, 1.0)
        #plt.ylim(0.0, 1.0)
        #coeff = np.polyfit(np.exp(pnts[:,0]), pnts[:,1], 1)
        #plt.plot(x, exp_model(coeff, x), color="green")

    #plt.plot(x, y, color="black")
    plt.plot(x, smoothed, color=colours[i+1])
    #plt.plot(x, y, color=colours[i+1])



    plt.xlabel("$M_{K_s}$")
    plt.ylabel("Luminosity Function (Arbitrary Units)")
    print("Plotted Branch ", i+1)

plt.plot(x, total, linestyle='-.', color='black')
coeff = np.trapz(total, x)
print("Integral: ", coeff)
plt.tight_layout()
plt.legend(['Red Giant Branch',
            'Red Giant Branch Bump',
            'Red Clump',
            'Asymptotic Giant Branch',
            'Alternative Method',
            'Total'
            ])

plt.show()
