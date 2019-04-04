import numpy as np
import matplotlib.pyplot as plt
from IO import readData
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

def nnfit():
    features,labels = readData("../data/train.csv")
    model = MLPClassifier(hidden_layer_sizes=(1000,1000),activation='tanh',solver='adam',alpha = 0.0001, batch_size=1000,learning_rate='invscaling',learning_rate_init=0.005,power_t=0.6,max_iter=300,shuffle=True,tol=1.0)
    print(cross_val_score(model,features,labels,cv=3))


if __name__ == "__main__":
    nnfit()
    






