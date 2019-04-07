import numpy as np
import matplotlib as mpl
import sys 
import matplotlib.pyplot as plt

def render(filename,i):
    print('running render')
    A = np.genfromtxt(filename,skip_header=1,dtype=float,delimiter=',')
    img = np.array(A[i,:],copy=True)
    print(img.shape)
    img = img.reshape(28,28)
    print(img.shape)
    plt.imshow(img,cmap='gray')
    plt.savefig("img" + str(i)+filename+"render"+ ".png")


if __name__ == '__main__':
    render(sys.argv[1],int(sys.argv[2]))
