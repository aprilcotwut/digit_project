from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import minmax_scale
from IO import readData
import os




if __name__=="__main__":
    parameters = {'n_estimators':[10,50,100,300],'min_samples_split':[2,4,8,16]}
    dt = RandomForestClassifier(verbose=10,max_features='sqrt')
    clf = GridSearchCV(dt,parameters,verbose = 10, n_jobs = 2,cv=4)
    X_pre,y = readData('../data/train.csv')
    X = minmax_scale(X_pre)
    clf.fit(X,y)
    joblib.dump(clf,"dt.joblib")

