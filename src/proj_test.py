import numpy as np
import joblib as jl
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import minmax_scale
from IO import readData
import os
import sys
import pandas as pds

def readModel(filename):
    clf = joblib.load(filename)
    return clf

def genCVplots(filename):
    clf = readModel(filename)
    dat = pds.DataFrame(clf.cv_results_)
    print(dat)
    #fig = dat.plot('param_alpha','mean_test_score',kind='line',subplots=True,logx=True)[0].get_figure()
    #fig.savefig("dat.png")
    #params = {'a':[0.5,0.1,0.001,0.0001],'hl':[(128,128),(512,),(256,)],'s':['sgd','adam']}
    
    hl128x128_solS = []
    hl128x128_solA = []
    hl256_solS = []
    hl256_solA = []
    hl512_solS = []
    hl512_solA = []
    perm = [hl128x128_solS,hl128x128_solA,hl256_solS,hl256_solA,hl512_solS,hl512_solA]
    for x in range(0,6):
        for y in range(0,4):
            i = x + 6*y
            perm[x].append([dat['param_alpha'][i],dat['mean_test_score'][i],dat['mean_train_score'][i],dat['mean_fit_time'][i]])
    
    #print(np.array(perm)[0,:,:])
    pl = np.array(perm)


    r = ['2 HL, 128 nodes','1 HL, 256 nodes','1 HL, 512 nodes']
    c = ['sgd','adam']
    for x in range(0,6):
        plt.subplot(3,2,x+1)
        plt.plot(-np.log10(pl[x,:,0]),pl[x,:,1])
        plt.xlabel(r'$-log_{10}(\alpha)$')
        plt.title(c[x%2])
        plt.ylabel(r[x%3])
    plt.tight_layout()
    plt.savefig("CVscoreVsAlpha.png")
    plt.close()

    for x in range(0,6):
        plt.subplot(3,2,x+1)
        plt.plot(-np.log10(pl[x,:,0]),pl[x,:,2])
        plt.xlabel(r'$-log_{10}(\alpha)$')
        plt.title(c[x%2])
        plt.ylabel(r[x%3])
    plt.tight_layout()
    plt.savefig("TrainscoreVsAlpha.png")
    plt.close()
    #pds.plotting.parallel_coordinates(dat,'Names')

def prediction(modelfile, datafile, outfile):
    clf = readModel(modelfile).best_estimator_
    X_t = readData(datafile,test=True)
    X = minmax_scale(X_t)

    y = clf.predict(X)
    x = np.array(range(1,y.shape[0]+1)).astype(int)
    print(x[0:10])
    print(y.shape)
    print(x.shape)
    np.savetxt(outfile,np.transpose(np.array([x,y])).astype(int),fmt='%i',delimiter=',',header="ImageId,Label",comments='')






if __name__ == "__main__":
        genCVplots(sys.argv[1])
        #prediction(sys.argv[1],'../data/test.csv','out.csv') 
