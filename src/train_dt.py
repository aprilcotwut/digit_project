from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import minmax_scale
from IO import readData
import os




if __name__=="__main__":
    # parameter grid for the grid search
    parameters = {'n_estimators':[10,50,100,300],'min_samples_split':[2,4,8,16]}

    # parameters set here are held constant in grid search
    dt = RandomForestClassifier(verbose=10,max_features='sqrt')

    # trian model on earlier defined 
    clf = GridSearchCV(dt,parameters,verbose = 10, n_jobs = 2,cv=4)

    # read and scale features and labels
    X_pre,y = readData('../data/train.csv')
    X = minmax_scale(X_pre)

    # train the model, takes much less time to train than the neural net
    clf.fit(X,y)

    # write file to python pickle: kind of a big file
    joblib.dump(clf,"dt.joblib")

