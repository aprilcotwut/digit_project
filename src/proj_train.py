import joblib as jl
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import minmax_scale
from IO import readData
import os

"""
    This file will train a neural net on the test data,
    on two cores. This takes a lot of time to train
"""


if __name__=="__main__":
    # parameter grid for GridSearchCV
    parameters = {'alpha':[0.5,0.1,0.001,0.0001],'hidden_layer_sizes':[(128,128),(256,),(512,)], 'solver':['sgd','adam']}
    
    # the parameters set here are held constant in the subsequent grid search
    nn = MLPClassifier(verbose=10,activation='relu',learning_rate='adaptive',tol=1e-4)

    # this trains several models on the above defined grid, and saves some data
    clf = GridSearchCV(nn,parameters,verbose = 10, n_jobs = 2,cv=4)

    # read data, and scale it
    X_pre,y = readData('../data/train.csv')
    X = minmax_scale(X_pre)

    # train the model
    clf.fit(X,y)

    # save the model to a pickle
    joblib.dump(clf,"clf.joblib")
