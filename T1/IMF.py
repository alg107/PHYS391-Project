import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
from scipy.integrate import simps
import scipy.stats as ss


def chabrier(m):
    return (0.158/(np.log(10)*m))*np.exp(-((np.log10(m)-np.log10(0.079))**2/(2*0.69**2)))

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


def IMF_sample(N=int(1e3)):
    A = 0.8
    sample_range = np.linspace(A, 1.1, N)
    chabrier_distr = chabrier_gen(name="chabrier", a=A)
    norm_constant = normalisation(sample_range)

    pdf = chabrier_distr.pdf(x = sample_range, const = norm_constant)
    cdf = chabrier_distr.cdf(x = sample_range, const = norm_constant)
    samples = chabrier_distr.rvs(const = norm_constant, size = N)
    return samples
