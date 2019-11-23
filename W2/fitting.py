import numpy as np
import AGModels
import matplotlib.pyplot as plt

def normal(x, m, s):
    return np.exp(-((x-m)**2/(2*s**2)))

def LF_model(params, m):
    A, m1, s1, B, m2, s2, C, m3, s3, D, E = params
    return A*normal(m,m1, s1) + B*normal(m, m2, s2) + C*normal(m, m3, s3) + D*np.exp(m) + E

def fit_data(x,y):
    data = np.column_stack((x,y,0.01*x**0))
    guess_params = (0.1, -2.85, 0.07, 0.9, -1.5, 0.06, 0.18, -1.0, 0.09, 0.112, 0.1)
    ps, uncs = AGModels.calc_params_NL(guess_params, data, LF_model)
    AGModels.plot_model(data, LF_model, ps)
    AGModels.present_params(ps, uncs, 4)
