import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import fitting
import AGModels


def reconstruct_LF(fname, bins):
    samples = np.load(fname)
    #samples = np.concatenate((samples, np.load("Samples/Magnitudes/sampledmagsN4000000.npy")))
    plt.figure()
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # Plotting the LF histogram
    final_bins = bins
    bins_seq = np.linspace(-3.5, 1.0, final_bins)
    plt.hist(samples, bins_seq, density=True, histtype='step')
    counts,bin_edges = np.histogram(samples, bins_seq, density=True)
    plt.ylim(0.0, 1.0)
    plt.xlim(-3.5, 1.0)

    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    err = np.random.rand(bin_centres.size)*100

    plt.xlabel("$M_{K_s}$")
    plt.ylabel("Luminosity Function (Arbitrary Units)")

    plt.figure()
    #plt.scatter(bin_centres, counts, marker="x", color="blue")

    # # Creating a nice spline to see the pattern
    # xnew = np.linspace(bin_centres.min(), bin_centres.max(), 1000) 
    # spl = make_interp_spline(bin_centres, counts, k=3)  # type: BSpline
    # power_smooth = spl(xnew)
    # plt.plot(xnew, power_smooth)

    plt.xlabel("$M_{K_s}$")
    plt.ylabel("Luminosity Function (Arbitrary Units)")



    return bin_centres, counts



if __name__ == "__main__":
    bcs, counts = reconstruct_LF("Samples/Magnitudes/sampledmagsN10000000.npy", 200)
    plt.scatter(bcs, counts, marker="+", color="black", s=20, alpha=0.5)
    fitting.fit_data(bcs, counts)
    plt.show()
