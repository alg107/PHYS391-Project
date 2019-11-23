import numpy as np
import matplotlib.pyplot as plt
import fitting
import AGModels


def reconstruct_LF(fname, bins):
    samples = np.load(fname)
    plt.figure()

    # Plotting the LF histogram
    final_bins = bins
    bins_seq = np.linspace(-3.5, 1.0, final_bins)
    plt.hist(samples, bins_seq, density=True, histtype='step')
    counts,bin_edges = np.histogram(samples, bins_seq, density=True)

    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    err = np.random.rand(bin_centres.size)*100

    plt.title("LF Histogram")
    plt.xlabel("Magnitude")
    plt.ylabel("Density")

    plt.figure()
    plt.scatter(bin_centres, counts, marker="x", color="black")

    # # Creating a nice spline to see the pattern
    # xnew = np.linspace(bin_centres.min(), bin_centres.max(), 1000) 
    # spl = make_interp_spline(bin_centres, counts, k=3)  # type: BSpline
    # power_smooth = spl(xnew)
    # plt.plot(xnew, power_smooth)

    plt.title("LF Scatter")
    plt.xlabel("Magnitude")
    plt.ylabel("Density")



    return bin_centres, counts



if __name__ == "__main__":
    bcs, counts = reconstruct_LF("Samples/Magnitudes/sampledmagsN1000000.npy", 200)
    fitting.fit_data(bcs, counts)
    plt.show()
