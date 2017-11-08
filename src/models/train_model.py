from macpath import norm_error
from sklearn.naive_bayes import GaussianNB
from settings import TRAINING_OPTIONS, MODELS_ROOT
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import read_training_processed_data
from sklearn.model_selection import LeaveOneGroupOut,  cross_val_score, cross_validate
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
import pickle
import logging
import os
import random
import numpy
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter



def instance_based( k):
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto')
    return model


def get_array(idx_array, x):
    return [x[i] for i in idx_array]

# def mean(numbers):
#     return float(sum(numbers)) / max(len(numbers), 1)

def main():
    # Read processed file
    X, y, groups = read_training_processed_data()

    numpy.set_printoptions(precision=3)
    # Train
    for option in TRAINING_OPTIONS[0:1]:
        max_score = numpy.asarray([0])

        if option == 'instance-based':
            # for k in range(1, 30, 5):
            #     lda = LinearDiscriminantAnalysis()
            #     lda.fit(X, y)
            #     X_r2 = lda.transform(X)
            #     model = instance_based(k)
            #     model.fit(X_r2, y)
            #     model2 = instance_based(k)
            #     current_score = cross_val_score(model2, X_r2, y, groups, cv = LeaveOneGroupOut(), scoring = 'accuracy')
            #     print k, current_score, current_score.mean()
            #     if current_score.mean() > max_score.mean():
            #         max_score = current_score
            #         pickle.dump(lda, open(os.path.join(MODELS_ROOT, option + '-lda' + '.model'), "wb"))
            #         pickle.dump(model, open(os.path.join(MODELS_ROOT, option + '.model'), "wb"))

            for k in range(1, 10, 1):
                logo = LeaveOneGroupOut()
                fold_scores = numpy.array([])
                model2 = KNeighborsClassifier(k)
                X_r2, y_r2 = SMOTE().fit_sample(X, y)
                scaler2 = preprocessing.MaxAbsScaler().fit(X_r2)
                X_r2 = scaler2.transform(X_r2)
                model2.fit(X_r2, y_r2)



                for train_index, test_index in logo.split(X, y, groups):
                    X_train, X_test = get_array(train_index, X), get_array(test_index, X)
                    y_train, y_test = get_array(train_index, y), get_array(test_index, y)

                    model = KNeighborsClassifier(k)

                    X_train, y_train = SMOTE().fit_sample(X_train, y_train)

                    scaler = preprocessing.MaxAbsScaler().fit(X_train)
                    X_train = scaler.transform(X_train)
                    X_test = scaler.transform(X_test)



                    # print(sorted(Counter(y_train).items()))



                    # lda = PCA()
                    # lda.fit(X_train, y_train)
                    # X_new = lda.transform(X_train)
                    # X_test_new = lda.transform(X_test)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    current_score = accuracy_score(y_pred, numpy.asarray(y_test))

                    # max_score = max(max_score, current_score)
                    fold_scores = numpy.append(fold_scores, [current_score])
                    # print y_pred.tolist()
                    # print y_test


                    y_pred2 = model.predict(X_train)
                    # print confusion_matrix(y_train, y_pred2)
                    # print accuracy_score(y_train, y_pred2).mean(), '\n'
                    #
                    # print confusion_matrix(y_test, y_pred)
                    # print "True:\t", y_train.tolist().count(0)
                    # print "False:\t", y_train.tolist().count(1)
                    # print "Unverified:\t", y_train.tolist().count(2)
                    # print "(Test)True:\t", y_test.count(0)
                    # print "(Test)False:\t", y_test.count(1)
                    # print "(Test)Unverified:\t", y_test.count(2), "\n"


                # print fold_scores, mean(fold_scores)
                print k, '\t\t', fold_scores, '\t\t', fold_scores.mean()

            if current_score.mean() > max_score.mean():
                max_score = current_score
                pickle.dump(scaler2, open(os.path.join(MODELS_ROOT, option + '-lda' + '.model'), "wb"))
                pickle.dump(model2, open(os.path.join(MODELS_ROOT, option + '.model'), "wb"))



            # logo = LeaveOneGroupOut()
            # for feed in (1, 7):
            #     fold_scores = []
            #     for train_index, test_index in logo.split(X, y, groups):
            #         X_train, X_test = get_array(train_index, X), get_array(test_index, X)
            #         y_train, y_test = get_array(train_index, y), get_array(test_index, y)
            #         n_test = len(test_index)
            #
            #         one_fold_score = []
            #         ss = ShuffleSplit(n_splits=3, test_size=0.4,random_state = random.randint(1,1000))
            #         for extra_train_index, test_index in ss.split(X_test):
            #             extra_train_index = extra_train_index[:(n_test*feed/10)]
            #             current_X_train = X_train + get_array(extra_train_index,X_test)
            #             current_y_train = y_train + get_array(extra_train_index,y_test)
            #             current_X_test = get_array(test_index, X_test)
            #             current_y_test = get_array(test_index, y_test)
            #
            #             max_score = 0
            #             for k in range(2,10,2):
            #                 model = instance_based(k)
            #                 lda = LinearDiscriminantAnalysis()
            #                 lda.fit(current_X_train, current_y_train)
            #                 X_new = lda.transform(current_X_train)
            #                 model.fit(X_new, current_y_train)
            #                 X_test_new = lda.transform(current_X_test)
            #                 y_pred = model.predict(X_test_new)
            #                 current_score = precision_score(y_pred, numpy.asarray(current_y_test), average = 'macro')
            #                 max_score = max(max_score, current_score)
            #
            #             one_fold_score.append(max_score)
            #
            #         fold_scores.append(mean(one_fold_score))
            #         # print fold_scores
            #
            #     print fold_scores, mean(fold_scores)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
