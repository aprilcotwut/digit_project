import pandas as pds
import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt

# skl expect n_samples x n_features array and n_samples array
def readData(filename):
    data = np.genfromtxt(filename,dtype=float,delimiter=",",skip_header=1)

    # shuffle rows of data array
    np.random.shuffle(data)

    
    labels = np.array(data[:,0],copy=True)
    features = np.array(data[:,1:])
    print(labels.shape)
    print(features.shape)

    return features, labels





if __name__ == "__main__":
    readData("../data/train.csv")