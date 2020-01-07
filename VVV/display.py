import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from matplotlib import colors
import matplotlib as mpl

mpl.rc('text', usetex=True)
LBLPAD = -2

def present_left(fig, ax, left_map, L_bins, B_bins, middle_map, right_map):

    middle_map = middle_map.T
    right_map = right_map.T

    # Getting dimensions
    # left, right, bottom, top
    extent = [L_bins[0].left, L_bins[-1].right, B_bins[0].left, B_bins[-1].right]

    # The plot has l and b on different axes than I initially chose
    # so we just transpose the plot arrays
    left_map = left_map.T

    # Display this array
    divnorm = colors.DivergingNorm(vmin=-0.07, vcenter=0.0, vmax=0.2)
    mpb = ax.imshow(left_map, cmap="RdBu_r", origin='lower', extent=extent,
            vmin=-0.07, vmax=0.2, norm=divnorm)

    fig.colorbar(mpb, orientation="horizontal",
            label="$\langle K_{s_{2MASS}}-K_{s_{VVV}} \\rangle (mag)$",
            ax=ax)
    
    ax.contour(middle_map, [0.9], colors="black", extent=extent,
            linestyles="dashed")
    ax.contour(right_map, [0.06], colors="black", extent=extent)


    ax.invert_xaxis()
    ax.set_xlabel("$l(^{\circ})$")
    ax.set_ylabel("$b(^{\circ})$", labelpad=LBLPAD)

    print("plotted left plot")


def present_middle(fig, ax, middle_map, L_bins, B_bins):

    extent = [L_bins[0].left, L_bins[-1].right, B_bins[0].left, B_bins[-1].right]

    middle_map = middle_map.T

    mpb = ax.imshow(middle_map, cmap='gnuplot', origin='lower', extent=extent,
            vmin=0.0, vmax=3.0)
    fig.colorbar(mpb, orientation="horizontal",
            label="$E(J-K_s)(mag)$",
            ax=ax)
    ax.contour(middle_map, [0.9], colors="white", extent=extent)
    ax.invert_xaxis()
    ax.set_xlabel("$l(^{\circ})$")
    ax.set_ylabel("$b(^{\circ})$", labelpad=LBLPAD)

    print("Plotted middle plot")


def present_right(fig, ax, right_map, L_bins, B_bins):

    # Getting dimensions
    # left, right, bottom, top
    extent = [L_bins[0].left, L_bins[-1].right, B_bins[0].left, B_bins[-1].right]

    # The plot has l and b on different axes than I initially chose
    # so we just transpose the plot arrays
    right_map = right_map.T

    # Display this array

    mpb = ax.imshow(right_map, cmap='gnuplot', origin='lower', extent=extent,
            vmin=0.0, vmax=0.18)
    fig.colorbar(mpb, orientation="horizontal", 
            label="$\langle \sigma_{K_s} \\rangle (mag)$", 
            ax=ax)
    ax.contour(right_map, [0.06], colors="white", extent=extent)
    ax.invert_xaxis()
    ax.set_xlabel("$l(^{\circ})$")
    ax.set_ylabel("$b(^{\circ})$", labelpad=LBLPAD)

    print("plotted right plot")


if __name__=="__main__":
    middle_map, right_map, left_map,  L_bins, B_bins = load_saved_maps()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.tight_layout()
    

    present_left(fig, ax1, left_map, L_bins, B_bins, middle_map, right_map)
    present_middle(fig, ax2, middle_map, L_bins, B_bins)
    present_right(fig, ax3, right_map, L_bins, B_bins)
    plt.show()
