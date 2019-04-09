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
    #print(dat)
     
    perm = [[],[],[],[]]
    for y in range(0,4):
        for x in range(0,4):
            i = x + 4*y
            perm[x].append([dat['param_min_samples_split'][i],dat['mean_test_score'][i],dat['mean_train_score'][i],dat['mean_fit_time'][i]])
    
    #print(np.array(perm)[0,:,:])
    pl = np.array(perm)

    lab = ['10 trees','50 trees', '100 trees', '300 trees']
    for x in range(0,4):
        plt.plot(np.log2(pl[x,:,0]),pl[x,:,1],label=lab[x])

    #plt.title(c[x%2])
    plt.legend()
    plt.ylabel("mean cross validation score")
    #plt.tight_layout()
    plt.xlabel(r'$log_{2}(min leaf size)$')
    plt.savefig("CVscoreVsLeafSize.png")
    plt.close()

    for x in range(0,4):
        plt.plot(np.log2(pl[x,:,0]),pl[x,:,2],label=lab[x])

    #plt.title(c[x%2])
    plt.legend()
    plt.ylabel("mean cross validation train score")
    #plt.tight_layout()
    plt.xlabel(r'$log_{2}(min leaf size)$')
    plt.savefig("CVtestVsLeafSize.png")
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
        prediction(sys.argv[1],'../data/test.csv','out_dt.csv') 

