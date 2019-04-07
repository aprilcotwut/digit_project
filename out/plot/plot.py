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
    a = np.log10(A[:5,2])
    b = A[:5,5]
    c = np.log10(A[5:,2])
    d = A[5:,5]
    plt.plot(a,b,'y-',label="1 hidden layer")
    plt.plot(c,d,'c-',label="3 hidden layers")
    plt.xlabel(r'$ log_{10} (\alpha )$')
    plt.ylabel("Mean cross validation score (5 folds)")
    plt.legend()
    plt.savefig("cvs.dat.png")


if __name__ == '__main__':
    plot(sys.argv[1])
