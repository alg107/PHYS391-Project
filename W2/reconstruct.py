import numpy as np
import matplotlib.pyplot as plt

arr = np.load("sampledmagsN1000.npy")

plt.figure()
final_bins = 100
bins_seq = np.linspace(-3.5, 1.0, final_bins)
plt.hist(arr, bins_seq, density=True, histtype='step')
plt.show()
