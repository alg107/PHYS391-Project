import numpy as np
import AGModels
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

RES = 1000

# S is now skewness
def normal(x, m, s):
    #return skewnorm.pdf(x-m, s)
    return np.exp(-((x-m)**2/(2*s**2)))

def LF_model(params, m):
    A, m1, s1, B, m2, s2, C, m3, s3, D, E = params
    return A*normal(m,m1, s1) + B*normal(m, m2, s2) + C*normal(m, m3, s3) + D*np.exp(m) + E

def plot_normal(data, A, m, s, color="black"):
    ts= np.array(data[:,0])
    ys = np.array(data[:,1])
    t_arr = np.linspace(np.min(ts), np.max(ts), RES)
    plt.plot(t_arr, A*normal(t_arr, m, s), color=color)

def plot_exp(data, D, E, color="black"):
    ts= np.array(data[:,0])
    ys = np.array(data[:,1])
    t_arr = np.linspace(np.min(ts), np.max(ts), RES)
    plt.plot(t_arr, D*np.exp(t_arr)+E, color=color)

def fit_data(x,y):
    data = np.column_stack((x,y,0.01*x**0))
    guess_params = (0.1, -2.85, 0.07, 0.9, -1.5, 0.06, 0.18, -1.0, 0.08, 0.112, 0.1)
    #guess_params = (0.1, -2.85, 0, 0.9, -1.5, 0, 0.18, -1.0, 0, 0.112, 0.1)
    ps, uncs = AGModels.calc_params_NL(guess_params, data, LF_model)
    plot_exp(data, ps[9], ps[10], color="green")
    plot_normal(data, ps[6], ps[7], ps[8], color="orange")
    plot_normal(data, ps[3], ps[4], ps[5], color="purple")
    plot_normal(data, ps[0], ps[1], ps[2], color="pink")
    AGModels.plot_model(data, LF_model, ps)
    #AGModels.plot_model(data, LF_model, guess_params)
    plt.ylim(0.0, 1.0)
    plt.xlim(-3.5, 1.0)
    plt.legend(("Red Giant Branch", "Red Giant Branch Bump", "Red Clump", "Asymptotic Giant Branch", "Total"))
    AGModels.present_params(ps, uncs, 4)
