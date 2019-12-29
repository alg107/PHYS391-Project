# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Final LF
#
# This is my final attempt at generating a luminosity function using a semi-analytic method.
#
# The general problem to be solved is that we have two random variables $X$ and $Y$ distributed normally and lognormally respectively. We have a third random variable that is given by $Z = f(X,Y)$ where f is an arbitrary function (the isochrone relation). We want to know how $Z$ is distributed. This can be done via a monte-carlo simulation or potentially by some semi-analytic method however past trials have shown the monte-carlo and SA methods differ in result. They are very similar but there is a noticeable difference in the exponential background.
#
# The method being trialed in this notebook will be different to the two method already trialed. We will call the final LF $p(M)$ and this is given by $\frac{dP(M)}{dM}$ where $P(M)$ is the CDF of $Z$. This CDF is given by $P(M^*) = \int\int_{D_{M^*}}p(m)p(z) dm dz$. $D_{M^*}$ is any $M \leq M^*$ 

# %%
# %load_ext autoreload
# %autoreload
# %matplotlib inline

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from Iso import Isochrone, chabrier, MDF
import sys
sys.path.insert(1, '../T1')
import reconstruct
import fitting

# %%
iso = Isochrone()

# This function represents the CDF of magnitude
def P(M):
    local_df = iso.df[iso.df.Kmag<=M]
    # Should return the above integral under the scatter plot of local_df
    # But I cannot figure out how this should be done
    #
    # The PDFs for mass and metallicity are the functions chabrier and MDF
    # respectively
    iso.plot(local_df)
P_V = np.vectorize(P)

P(-2.0)

# lowercase p represents the PDF of magnitude
def calc_p(x):
    Ps = P_V(x)
    P_spl = UnivariateSpline(x, Ps)
    p_spl = P_spl.derivative()
    ps = np.array([p_spl(i) for i in x])
    return ps

def plot_p(x, ps):
    plt.figure()
    plt.plot(x, ps, color="blue")
    plt.xlabel("$M_{K_s}$")
    plt.ylabel("Luminosity Function (Arbitrary Units)")

x = np.linspace(-3.5, 1.0, 1000)
#plot_p(x, calc_p(x))

plt.show()

# %% [markdown]
# ## Monte-Carlo Simulation
#
# The following code block presents the results of my latest monte-carlo simulation for comparison.

# %%
bcs, counts = reconstruct.reconstruct_LF("/users/alex/Data/MagSamples/SampledMags.npy", 200)

print("Gaussian Fit\n")

# Plot the histogram bins as points on a scatter plot
plt.scatter(bcs, counts, marker="+", color="black", s=20, alpha=0.5)

# Calls the fit_data routine in fitting.py which is quite involved
# This one is for fitting gaussians
fitting.fit_data(bcs, counts)

plt.figure()
plt.xlabel("$M_{K_s}$")
plt.ylabel("Luminosity Function (Arbitrary Units)")

# The same thing again for the skew fit
plt.scatter(bcs, counts, marker="+", color="black", s=20, alpha=0.5)

print("\nSkew Fit\n")
fitting.fit_data_skew(bcs, counts)
print("Disregard uncertainties for now.")
plt.show()

# %%
