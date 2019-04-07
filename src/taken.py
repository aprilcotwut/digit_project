import numpy as np
from scipy.ndimage import convolve
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.externals import joblib
import os.path
from sklearn.preprocessing import minmax_scale
from IO import readData

PATH = 'mlp_model.pkl'

if __name__ == '__main__':
    #print('Fetching and loading MNIST data')
    #mnist = fetch_mldata('MNIST original')
    #X, y = mnist.data, mnist.target
    #X_train, X_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.25)

    #print('Got MNIST with %d training- and %d test samples' % (len(y_train), len(y_test)))
    #print('Digit distribution in whole dataset:', np.bincount(y.astype('int64')))
    X, y_train = readData('../data/train.csv')
    X_train = minmax_scale(X)
    #clf = None
    #if os.path.exists(PATH):
    #    print('Loading model from file.')
    #    clf = joblib.load(PATH).best_estimator_
    #else:
    #print('Training model.')
    #params = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
    mlp = MLPClassifier(verbose=10,hidden_layer_sizes=(512,) ,learning_rate='adaptive')
    print(cross_val_score(mlp,X_train,y_train,cv=5))
    #clf = GridSearchCV(mlp, params, verbose=10, n_jobs=-1, cv=5)
    #clf.fit(X_train, y_train)
    #print('Finished with grid search with best mean cross-validated score:', clf.best_score_)
    #print('Best params appeared to be', clf.best_params_)
    #joblib.dump(clf, PATH)
    #clf = clf.best_estimator_

#print('Test accuracy:', clf.score(X_test, y_test))
"""
(42000,)
(42000, 784)
Iteration 1, loss = 0.38702103
Iteration 2, loss = 0.16765635
Iteration 3, loss = 0.11730479
Iteration 4, loss = 0.08388761
Iteration 5, loss = 0.06552971
Iteration 6, loss = 0.04904557
Iteration 7, loss = 0.03886303
Iteration 8, loss = 0.03045402
Iteration 9, loss = 0.02475195
Iteration 10, loss = 0.01936372
Iteration 11, loss = 0.01478687
Iteration 12, loss = 0.01178684
Iteration 13, loss = 0.01077100
Iteration 14, loss = 0.00783104
Iteration 15, loss = 0.00605363
Iteration 16, loss = 0.00484032
Iteration 17, loss = 0.00408994
Iteration 18, loss = 0.00379024
Iteration 19, loss = 0.00293620
Iteration 20, loss = 0.00246311
Iteration 21, loss = 0.00215111
Iteration 22, loss = 0.00193898
Iteration 23, loss = 0.00176680
Iteration 24, loss = 0.00159807
Iteration 25, loss = 0.00147985
Iteration 26, loss = 0.00133139
Iteration 27, loss = 0.00123555
Iteration 28, loss = 0.00122796
Iteration 29, loss = 0.00119713
Training loss did not improve more than tol=0.000100 for two consecutive epochs. Stopping.
Iteration 1, loss = 0.39194122
Iteration 2, loss = 0.16753445
Iteration 3, loss = 0.11419861
Iteration 4, loss = 0.08459560
Iteration 5, loss = 0.06318389
Iteration 6, loss = 0.04941329
Iteration 7, loss = 0.03777521
Iteration 8, loss = 0.02994131
Iteration 9, loss = 0.02197493
Iteration 10, loss = 0.01799536
Iteration 11, loss = 0.01416025
Iteration 12, loss = 0.01072992
Iteration 13, loss = 0.00826726
Iteration 14, loss = 0.00649703
Iteration 15, loss = 0.00556540
Iteration 16, loss = 0.00435483
Iteration 17, loss = 0.00373953
Iteration 18, loss = 0.00360576
Iteration 19, loss = 0.00281014
Iteration 20, loss = 0.00223228
Iteration 21, loss = 0.00200727
Iteration 22, loss = 0.00181457
Iteration 23, loss = 0.00165947
Iteration 24, loss = 0.00152215
Iteration 25, loss = 0.00140071
Iteration 26, loss = 0.00128393
Iteration 27, loss = 0.00120808
Iteration 28, loss = 0.00111263
Iteration 29, loss = 0.00104442
Training loss did not improve more than tol=0.000100 for two consecutive epochs. Stopping.
Iteration 1, loss = 0.38195700
Iteration 2, loss = 0.16274892
Iteration 3, loss = 0.11635078
Iteration 4, loss = 0.08346899
Iteration 5, loss = 0.06184884
Iteration 6, loss = 0.04840645
Iteration 7, loss = 0.04358516
Iteration 8, loss = 0.04098652
Iteration 9, loss = 0.02443173
Iteration 10, loss = 0.01929429
Iteration 11, loss = 0.01469890
Iteration 12, loss = 0.01171685
Iteration 13, loss = 0.00906175
Iteration 14, loss = 0.00773817
Iteration 15, loss = 0.00618602
Iteration 16, loss = 0.00500838
Iteration 17, loss = 0.00434320
Iteration 18, loss = 0.00368005
Iteration 19, loss = 0.00317087
Iteration 20, loss = 0.00281439
Iteration 21, loss = 0.00242538
Iteration 22, loss = 0.00217659
Iteration 23, loss = 0.00462022
Iteration 24, loss = 0.00341805
Iteration 25, loss = 0.01922347
Training loss did not improve more than tol=0.000100 for two consecutive epochs. Stopping.
Iteration 1, loss = 0.38347602
Iteration 2, loss = 0.16880226
Iteration 3, loss = 0.11650212
Iteration 4, loss = 0.08656777
Iteration 5, loss = 0.06588350
Iteration 6, loss = 0.04929263
Iteration 7, loss = 0.03859818
Iteration 8, loss = 0.03091436
Iteration 9, loss = 0.02419425
Iteration 10, loss = 0.01986656
Iteration 11, loss = 0.01429560
Iteration 12, loss = 0.01255083
Iteration 13, loss = 0.00953616
Iteration 14, loss = 0.00718936
Iteration 15, loss = 0.00648222
Iteration 16, loss = 0.00532448
Iteration 17, loss = 0.00433760
Iteration 18, loss = 0.00348636
Iteration 19, loss = 0.00285801
Iteration 20, loss = 0.00290444
Iteration 21, loss = 0.00222662
Iteration 22, loss = 0.00207893
Iteration 23, loss = 0.00222624
Iteration 24, loss = 0.00170104
Iteration 25, loss = 0.00148507
Iteration 26, loss = 0.00134133
Iteration 27, loss = 0.00124669
Iteration 28, loss = 0.00117972
Iteration 29, loss = 0.00109029
Training loss did not improve more than tol=0.000100 for two consecutive epochs. Stopping.
Iteration 1, loss = 0.38560267
Iteration 2, loss = 0.16754952
Iteration 3, loss = 0.11491132
Iteration 4, loss = 0.08377139
Iteration 5, loss = 0.08164395
Iteration 6, loss = 0.05131074
Iteration 7, loss = 0.04088621
Iteration 8, loss = 0.03164100
Iteration 9, loss = 0.02607010
Iteration 10, loss = 0.02010530
Iteration 11, loss = 0.01583448
Iteration 12, loss = 0.01281293
Iteration 13, loss = 0.01044561
Iteration 14, loss = 0.00845797
Iteration 15, loss = 0.00804568
Iteration 16, loss = 0.00544132
Iteration 17, loss = 0.00448443
Iteration 18, loss = 0.00379756
Iteration 19, loss = 0.00343144
Iteration 20, loss = 0.00284303
Iteration 21, loss = 0.00679655
Iteration 22, loss = 0.01047298
Iteration 23, loss = 0.00377163
Training loss did not improve more than tol=0.000100 for two consecutive epochs. Stopping.
[ 0.97763236  0.9791741   0.97261579  0.97653924  0.97665555]
"""
