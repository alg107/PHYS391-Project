import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, lognorm
from scipy.integrate import simps, quad
import scipy.stats as ss
from scipy.interpolate import UnivariateSpline, interp1d


def chabrier(m):
    return (0.158/(np.log(10)*m))*np.exp(-((np.log10(m)-np.log10(0.079))**2/(2*0.69**2)))

def chabrier2(m):
    if m > 0.8 and m < 1.1:
        return (1/0.0019267)*np.exp(-(716.4/m)**0.25)*m**-(3.3)
    else:
        return 0.0
chabrier2_V = np.vectorize(chabrier2)

def calc_cdf(pdf, xmin, xmax):
    x = np.linspace(xmin, xmax, 1000)
    cdfs = np.array([quad(pdf, 0.0, i)[0] for i in x])
    return x, cdfs

def normalisation(x):
	return simps(chabrier(x), x)

class chabrier_gen(ss.rv_continuous): 
    def _pdf(self, x, const):
        return (1.0/const) * chabrier(x)

def load_samples():
    sfiles = [
            "1000000",
            # "4000000",
            # "5000000"
            ]
    samples = np.array([])
    for s in sfiles:
        samples = np.concatenate((samples, np.load("Samples/IMF/sampledIMF_N"+s+".npy")))
    print(samples.shape)
    return samples


# def IMF_sample(N=int(1e3)):
    # A = 0.8
    # sample_range = np.linspace(A, 1.1, N)
    # chabrier_distr = chabrier_gen(name="chabrier", a=A)
    # norm_constant = normalisation(sample_range)

    # pdf = chabrier_distr.pdf(x = sample_range, const = norm_constant)
    # cdf = chabrier_distr.cdf(x = sample_range, const = norm_constant)
    # samples = chabrier_distr.rvs(const = norm_constant, size = N)
    # return samples

# def IMF_sample(N=int(1e3)):
    # mmin = 0.8
    # mmax = 1.1
    # distr = lognorm(0.57*np.log(10), 0.22)
    # samples = np.array([])
    # while len(samples)<N:
        # toAdd = distr.rvs(N)
        # toAdd = toAdd[(toAdd > mmin) & (toAdd < mmax)]
        # samples = np.append(samples, toAdd)
    # return samples[:N]

def IMF_sample(N=int(1e3)):
    xs, cdfs = calc_cdf(chabrier2_V, 0.8, 1.1)
    inv_cdf = interp1d(cdfs, xs)

    flat_samples = np.random.rand(N)
    mag_samples = inv_cdf(flat_samples)
    return mag_samples


if __name__=="__main__":
    N = int(1e6)
    sampled_IMF = IMF_sample2(N)
    IMF_bins = 20


    plt.figure()
    plt.title("IMF Distribution")
    IMF_seq = np.linspace(0.8, 1.1, IMF_bins)
    plt.hist(sampled_IMF, IMF_seq, density=True, histtype='step')
    X = np.linspace(0.8, 1.1, 1000)
    plt.plot(X, chabrier2_V(X))
    plt.show()
