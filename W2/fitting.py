import numpy as np
import AGModels
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, norm
from scipy.special import erf

"""
Does a lot of the work of fitting and plotting various models etc
"""

# Resolution for plots
RES = 1000

# Normal distribution PDF
def normal(x, m, s):
    return np.exp(-((x-m)**2/(2*s**2)))
    #return np.exp(-((x-m)**2/2))*(1+erf(s*(x-m)/(np.sqrt(2))))

# a is skewness
# Skewnormal distribution PDF
def skewnormal(x, m, s, a):
    return np.exp(-((x-m)**2/(2*s**2)))*(1+erf(a*(x-m)/(np.sqrt(2))))


# Gaussian LF model to be fitted
def LF_model(params, m):
    A, m1, s1, B, m2, s2, C, m3, s3, D, E = params
    return A*normal(m,m1, s1) + B*normal(m, m2, s2) + C*normal(m, m3, s3) + D*np.exp(m) + E

# Skew Gaussian LF model to be fitted
def LF_model_skew(params, m):
    A, m1, s1, a1, B, m2, s2, a2, C, m3, s3, a3, D, E = params
    return A*skewnormal(m,m1, s1, a1) + B*skewnormal(m, m2, s2, a2) + C*skewnormal(m, m3, s3, a3) + D*np.exp(m) + E


# Helper function to plot a gaussian
# Note that all these helper function find the min and max t vals based on
# the data
def plot_normal(data, A, m, s, color="black"):
    ts= np.array(data[:,0])
    ys = np.array(data[:,1])
    t_arr = np.linspace(np.min(ts), np.max(ts), RES)
    plt.plot(t_arr, A*normal(t_arr, m, s), color=color)

# Helper function to plot a skew gaussian
def plot_skew_normal(data, A, m, s, a, color="black"):
    ts= np.array(data[:,0])
    ys = np.array(data[:,1])
    t_arr = np.linspace(np.min(ts), np.max(ts), RES)
    plt.plot(t_arr, A*skewnormal(t_arr, m, s, a), color=color)

# Helper function to plot an exponential
def plot_exp(data, D, E, color="black"):
    ts= np.array(data[:,0])
    ys = np.array(data[:,1])
    t_arr = np.linspace(np.min(ts), np.max(ts), RES)
    plt.plot(t_arr, D*np.exp(t_arr)+E, color=color)

def setup_scatter_plot(title, ps, uncs):
    plt.ylim(0.0, 1.0)
    plt.xlim(-3.5, 1.0)
    plt.legend(("Red Giant Branch", "Red Giant Branch Bump", "Red Clump", "Asymptotic Giant Branch", "Total"))
    plt.title(title)
    AGModels.present_params(ps, uncs, 4)


arbitrary_unc_coeff = 0.01

# Uses my beautiful non-linear fitting library to fit the gaussian based
# LF model to the data given some guess parameters
def fit_data(x,y):
    data = np.column_stack((x,y,arbitrary_unc_coeff*x**0))
    guess_params = (0.1, -2.85, 0.07, 0.9, -1.5, 0.06, 0.18, -1.0, 0.08, 0.112, 0.1)
    ps, uncs = AGModels.calc_params_NL(guess_params, data, LF_model)
    #ps=guess_params
    plot_exp(data, ps[9], ps[10], color="green")
    plot_normal(data, ps[6], ps[7], ps[8], color="orange")
    plot_normal(data, ps[3], ps[4], ps[5], color="purple")
    plot_normal(data, ps[0], ps[1], ps[2], color="pink")
    AGModels.plot_model(data, LF_model, ps)
    setup_scatter_plot("Gaussian Fit", ps, uncs)

# Same thing but for a skew gaussian
def fit_data_skew(x, y):
    data = np.column_stack((x,y,arbitrary_unc_coeff*x**0))
    guess_params = (0.05, -2.85, 0.06, 1.0,
                    0.9, -1.5, 0.06, 1.0,
                    0.15, -1.0, 0.08, 1.0,
                    0.112, 0.1)
    ps, uncs = AGModels.calc_params_NL(guess_params, data, LF_model_skew)
    #ps=guess_params
    plot_exp(data, ps[12], ps[13], color="green")
    plot_skew_normal(data, ps[8], ps[9], ps[10], ps[11], color="orange")
    plot_skew_normal(data, ps[4], ps[5], ps[6], ps[7], color="purple")
    plot_skew_normal(data, ps[0], ps[1], ps[2], ps[3], color="pink")
    AGModels.plot_model(data, LF_model_skew, ps)
    setup_scatter_plot("Skew Fit", ps, uncs)
