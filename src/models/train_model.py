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
import numpy as np
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from copy import deepcopy
from settings import FEATURE_OPTIONS
from src.models import feature_bitmask
from utils import get_subset_features, get_array



def main():
    X, y, groups = read_training_processed_data()
    np.set_printoptions(precision=4)

    features_subset = 'social_features'

    balancing_class_algorithm = None
    scale_option = None
    reduce_dimension_algorithm = None

    training_algorithm = {
        'name': 'instance-based',
        'k': 1,
    }

    # balancing_class_algorithm = {
    #     'name': 'SMOTE',
    #     'k': 1,
    # }
    #
    # scale_option = {
    #     'name': 'MaxAbs'
    # }
    #
    # reduce_dimension_algorithm = {
    #     'name': 'PCA',
    #     'n_components': 100,
    # }

    train(
        X = X,
        y = y,
        groups = groups,
        algo_option = training_algorithm,
        feature_option = features_subset,
        balancing_option = balancing_class_algorithm,
        scale_option = scale_option,
        reduce_dimension_option = reduce_dimension_algorithm,
    )

def init_model(algo_option):
    if algo_option['name'] == 'instance-based':
        return KNeighborsClassifier(n_neighbors = algo_option['k'])

    return None


def init_balancing_model(balancing_option):
    if balancing_option['name'] == 'SMOTE':
        return SMOTE(k_neighbors = balancing_option['k'])

    return None


def init_scaler(scale_option):
    if scale_option['name'] == 'MaxAbs':
        return preprocessing.MaxAbsScaler()

    return None

def init_reduce_dimension_model(reduce_dimension_option):
    if reduce_dimension_option['name'] == 'PCA':
        return PCA(n_components=reduce_dimension_option['n_components'])

    return None


def train(X ,y, groups, algo_option, feature_option, balancing_option, scale_option, reduce_dimension_option):
    # Read processed file

    X_subset = get_subset_features(X, feature_option)
    y_subset = deepcopy(y)

    logo = LeaveOneGroupOut()
    fold_scores = np.zeros(5)

    # 5 folds corresponding to 5 events

    for train_index, test_index in logo.split(X_subset, y_subset, groups):

        # Split train and test from folds
        X_train, X_test = get_array(train_index, X), get_array(test_index, X)
        y_train, y_test = get_array(train_index, y), get_array(test_index, y)

        # Init a classifer
        model = init_model(algo_option)

        # Init an optional scaler
        if scale_option:
            scaler = init_scaler(scale_option)
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        # Init an optional balancing model
        if balancing_option:
            balancer = init_balancing_model(balancing_option)
            X_train, y_train = balancer.fit_sample(X_train, y_train)

        # Init an optional reduce dimenstion model
        if reduce_dimension_option:
            reducer = init_reduce_dimension_model(reduce_dimension_option)
            reducer.fit(X_train, y_train)
            X_train = reducer.transform(X_train)
            X_test = reducer.transform(X_test)

        # Fit prerocessed data to classifer model
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        current_score = accuracy_score(y_pred, np.asarray(y_test))

        # max_score = max(max_score, current_score)
        fold_scores = np.append(fold_scores, [current_score])
        # print y_pred.tolist()
        # print y_test


        # y_pred2 = model.predict(X_train)
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



    # # if fold_scores.mean() > max_score.mean():
    # max_score = fold_scores
    # pickle.dump(lda2, open(os.path.join(MODELS_ROOT, option + '-'+ str(k) +  '-lda' + '.model'), "wb"))
    # pickle.dump(scaler2, open(os.path.join(MODELS_ROOT, option + '-' + str(k) + '-scaler' + '.model'), "wb"))
    # pickle.dump(model2, open(os.path.join(MODELS_ROOT, option + '-'+ str(k) + '.model'), "wb"))
    #
    #
    # model2 = KNeighborsClassifier(1)
    #
    # X_r2 = deepcopy(X)
    # y_r2 = deepcopy(y)
    #
    # scaler2 = preprocessing.MaxAbsScaler().fit(X_r2)
    # X_r2 = scaler2.transform(X_r2)
    #
    # X_r2, y_r2 = SMOTE(k_neighbors=1).fit_sample(X_r2, y_r2)
    #
    # lda2 = PCA(n_components=300)
    # lda2.fit(X_r2, y_r2)
    # X_r2 = lda2.transform(X_r2)
    #
    # model2.fit(X_r2, y_r2)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
