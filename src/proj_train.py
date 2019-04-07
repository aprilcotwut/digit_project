import joblib as jl
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import minmax_scale
from IO import readData
import os




if __name__=="__main__":
    parameters = {'alpha':[0.5,0.1,0.001,0.0001],'hidden_layer_sizes':[(128,128),(256,),(512,)], 'solver':['sgd','adam']}
    nn = MLPClassifier(verbose=10,activation='relu',learning_rate='adaptive',tol=1e-4)
    clf = GridSearchCV(nn,parameters,verbose = 10, n_jobs = 2,cv=4)
    X_pre,y = readData('../data/train.csv')
    X = minmax_scale(X_pre)
    clf.fit(X,y)
    joblib.dump(clf,"clf.joblib")
