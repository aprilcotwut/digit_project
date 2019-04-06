import numpy as np
import sys
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from IO import readData
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale 

def nnfit(numberInHL,numberOfHl,activation,alpha=0.0001,tol=1.0,iters=100):
    features,labels = readData("../data/train.csv")
    feat_n =  minmax_scale(features)

    # alpha: regularization
    # activation: activation function, mostly using Tanh (arbitrarily)
    # batch_size: convergence speed, nothing to do with fitting
    # max_iter: mostly about time of training
    # you can probably get at least a page out of defining the learning-rate convention
    HL = tuple( numberOfHl*[numberInHL])
    models = []
    models.append(MLPClassifier(hidden_layer_sizes=HL,activation=activation,solver='sgd',alpha = alpha, batch_size=100,learning_rate='invscaling',learning_rate_init=0.01,power_t=0.6,max_iter=1,shuffle=True,tol=tol,warm_start=True))
    
    models.append(MLPClassifier(hidden_layer_sizes=HL,activation=activation,solver='sgd',alpha = alpha, batch_size=100,learning_rate='invscaling',learning_rate_init=0.001,power_t=0.6,max_iter=1,shuffle=True,tol=tol,warm_start=True))


    lossArr = [[],[]]
    index = 0
    for x in range(0,iters):
        index = index + 1
        models[0].fit(feat_n,labels)
        models[1].fit(feat_n,labels)
        lossArr[0].append(models[0].loss_)
        lossArr[1].append(models[1].loss_)
        #print(model.loss_)
   
    lossPlot0 = np.array(lossArr[0])
    lossPlot1 = np.array(lossArr[1])
    plt.plot(np.array(range(1,index +1)),lossPlot1,label="one")
    plt.plot(np.array(range(1,index+1)),lossPlot0,label="two")
    
    plt.xlabel("loss vs. number of epochs")
    plt.ylabel("loss")
    plt.xlabel("number of epochs")
    plt.savefig("testingLRI")

    #print(cross_val_score(model,features,labels,cv=3))

def learningRateScore(layers,nodes,alpha,tol,lri):

    features,labels = readData("../data/train.csv")
    feat_n =  minmax_scale(features)

    hl = tuple(layers*[nodes])
    model = MLPClassifier(hidden_layer_sizes=hl,activation='tanh', solver='sgd',alpha=alpha,batch_size=1000,learning_rate='invscaling',learning_rate_init=lri,power_t = 0.6,max_iter=100,shuffle=True,tol=tol)
    crossVal = cross_val_score(model,features,labels,cv=5)
    
    print(crossVal)
    fn = str(layers) + "l"
    fn = fn + str(nodes) + "n"
    fn = fn + str(alpha) + "a"
    fn = fn + str(tol) + "t"
    fn = fn + str(lri) + "l"
    fn = fn + ".txt"
    with open(fn,"w") as f:
        f.write(str(crossVal))

if __name__ == "__main__":
    layers = int(sys.argv[1])
    nodes = int(sys.argv[2])
    alpha = float(sys.argv[3])
    tol = float(sys.argv[4])
    lri = float(sys.argv[5])
    learningRateScore(layers,nodes,alpha,tol,lri)
    #learningRateScore(1,1000,0.000001,.01,0.01)
    #nnfit(50,2,'tanh')
    






