import numpy as np
import matplotlib.pyplot as plt
from helpers import *


def present_middle_right(middle_map, right_map, L_bins, B_bins):

    # Getting dimensions
    # left, right, bottom, top
    extent = [L_bins[0].left, L_bins[-1].right, B_bins[0].left, B_bins[-1].right]

    # The plot has l and b on different axes than I initially chose
    # so we just transpose the plot arrays
    middle_map = middle_map.T
    right_map = right_map.T

    # Display this array
    plt.figure()
    plt.imshow(middle_map, cmap='gnuplot', origin='lower', extent=extent,
            vmin=0.0, vmax=3.0)
    plt.colorbar(orientation="horizontal",
            label="$E(J-K_s)(mag)$")
    plt.contour(middle_map, [0.9], colors="white", extent=extent)
    plt.gca().invert_xaxis()
    plt.xlabel("$l(^{\circ})$")
    plt.ylabel("$b(^{\circ})$")

    plt.figure()
    plt.imshow(right_map, cmap='gnuplot', origin='lower', extent=extent,
            vmin=0.0, vmax=0.18)
    plt.colorbar(orientation="horizontal", 
            label="$\langle \sigma_{K_s} \\rangle (mag)$")
    plt.contour(right_map, [0.06], colors="white", extent=extent)
    plt.gca().invert_xaxis()
    plt.xlabel("$l(^{\circ})$")
    plt.ylabel("$b(^{\circ})$")

    print("plotted both plots")

    plt.show()


if __name__=="__main__":
    middle_map, right_map, L_bins, B_bins = load_saved_maps()
    present_middle_right(middle_map, right_map, L_bins, B_bins)
