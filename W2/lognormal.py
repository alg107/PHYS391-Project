import numpy as np
import matplotlib.pyplot as plt

def lnnormal(x, m, s):
    return (1/(s*x*np.sqrt(2*np.pi)))*np.exp(-((np.log(x)-m)**2)/(2*s**2))

def lognormal(x, m, s):
    return (1/(s*x*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x)-m)**2)/(2*s**2))


x = np.linspace(0.1, 10, 10000)

plt.plot(x, lnnormal(x, 0, 1))
plt.plot(x, lognormal(x, 0, 1))
plt.show()
