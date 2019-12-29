import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar as pb
from Iso import Isochrone
from scipy.integrate import simps

def CDF_Z(Kmag, iso):
    pass



if __name__ == "__main__":
    iso = Isochrone()
    iso.plot()

    print("3D isochrone plot done")

    iso.colour_plot()
    iso.plot_slice(-0.5)

    print("Colour plot done")

    plt.show()


