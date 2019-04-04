import matplotlib.pyplot as plt
import numpy as np
from IO import readData

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def main():
    clf = DecisionTreeClassifier(min_samples_split=10,min_samples_leaf=10)
    features, labels = readData("../data/train.csv")
    #clf.train(features,labels)
    print(cross_val_score(clf,features,labels,cv=5))
    



if __name__ == "__main__":    
    main()
