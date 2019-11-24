import numpy as np
import matplotlib.pyplot as plt
import pandas as pds
from scipy.optimize import minimize
from decimal import Decimal
import random

"""
TODO:
    - Add equivalent linear functions 
"""

RES = 400

def chisquare(obs, exp, error):
    chisqr = 0
    for i in range(len(obs)):
        chisqr = chisqr + ((obs[i]-exp[i])**2)/(error[i]**2)
    return chisqr

def chi2(params, data, fn):
    """Wrapper around chisquare function"""

    t = data[:,0]
    y = data[:,1]
    err = data[:,2]
    
    expected = fn(params, t)
    return chisquare(y, expected, err)

def process_file(filename):
    data = np.loadtxt(filename)
    return data

def plot_data(data, fignum, xlabel, ylabel):
    t= np.array(data[:,0])
    y = np.array(data[:,1])
    unc = np.array(data[:,2])

    plt.rcdefaults() # turn off xkcd
    plt.errorbar(t,y,unc,fmt='.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_model(data, fn, params):
    ts= np.array(data[:,0])
    ys = np.array(data[:,1])
    t_arr = np.linspace(np.min(ts), np.max(ts), RES)
    plt.plot(t_arr, fn(params, t_arr), color="blue", linestyle="--")

def plot_model_data(data, fn, params):
    ts= np.array(data[:,0])
    ys = np.array(data[:,1])
    t_arr = np.linspace(np.min(ts), np.max(ts), RES)
    plot_data(data, 1, "x", "y")
    plt.plot(t_arr, fn(params, t_arr))

"""
(1) Gives the params, given guesses, data and a function
    - Non-Linear
"""
def calc_params_NL(guesses, pnts, fn):
    result = minimize(chi2,guesses, method='BFGS',args=(pnts, fn))
    param_unc = np.sqrt(2*np.diag(result.hess_inv))
    newparams = result.x
    return newparams, param_unc


"""
(2) Fits and plots automatically and returns params
    - Non-Linear
"""
def proto_fit_NL(datap, fn, guess_params, figoptions=False):
    if isinstance(datap, str):
        data = process_file(datap)
    else:
        data = datap
    params, p_uncs = calc_params_NL(guess_params, data, fn)
    plt.figure()
    if figoptions:
        figoptions()
    plot_model_data(data, fn, params)
    #plt.show()
    return params, p_uncs 

def dummy_data(fn, params, i, f, N, unc=0.05, jitter=0.1):
    t = np.linspace(i,f, N*5).tolist()
    t = np.array(random.sample(t, N))
    y = fn(params, t)
    y = [a*random.uniform(1-jitter, 1+jitter) for a in y]
    uncs = np.array([random.uniform(-unc, unc)*a for a in y])
    return np.column_stack((t,y,uncs))

# Rounds uncertainties to one significant figure
def round_to_1(x):
    return round(x, -int(np.floor(np.log10(abs(x)))))

def present_params(ps, uncs, rounding):
    if ps.size != uncs.size:
        raise Exception("Parameters and uncertainties must have same size")
    print("Fitted Parameters:")
    for i,p in enumerate(uncs):
        perc = 100*uncs[i]/ps[i]
        print("Parameter "+ str(i+1) +": " ,end="")
        print(round(ps[i], rounding), "±", round_to_1(uncs[i]), "("+str(round(perc,2))+"%)")

def system_test(fn, params, i, f, guess_params):
    data2 = dummy_data(fn, params, i, f, RES)
    ps, uncs = proto_fit_NL(data2, fn, guess_params)
    present_params(ps, uncs, 4)
    return ps, uncs
    



