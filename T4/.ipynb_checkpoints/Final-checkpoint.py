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
