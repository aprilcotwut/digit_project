import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import sys
import numpy as np

def plot(filename):
    """
        This will take in the csv generated from
        the routine in ../scripts and plot
        crossval score vs alpha value
    """
    A = np.genfromtxt(filename,delimiter=',')
    print(A.shape)
    a = np.log10(A[:,2])
    b = A[:,5]
    plt.plot(a,b,'bs')
    plt.savefig("cvs.dat.png")


if __name__ == '__main__':
    plot(sys.argv[1])
