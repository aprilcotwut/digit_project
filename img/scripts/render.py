import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import sys 
import matplotlib.pyplot as plt

def render(filename,i):
    """
        Use this to take an example from test.csv and turn 
        that into a png image. Renders on run, but if you
        run over ssh, then you should comment out line
        19.
    """
    print('running render')
    A = np.genfromtxt(filename,skip_header=1,dtype=float,delimiter=',')
    img = np.array(A[i,:],copy=True)
    print(img.shape)
    img = img.reshape(28,28)
    img = 255 - img
    print(img.shape)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.savefig("img" + str(i)+"render"+ ".png")


if __name__ == '__main__':
    render(sys.argv[1],int(sys.argv[2]))
