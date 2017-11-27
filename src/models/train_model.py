from macpath import norm_error
from sklearn.naive_bayes import GaussianNB
from settings import *
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
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import *
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
import shutil


def main():
    X, y, groups = read_training_processed_data()
    np.set_printoptions(precision=4)

    features_subset = TRAINING_SETTINGS['features_subset']
    balancing_class_algorithm = TRAINING_SETTINGS['balancing_class_algorithm']
    scale_option = TRAINING_SETTINGS['scale_option']
    reduce_dimension_algorithm = TRAINING_SETTINGS['reduce_dimension_algorithm']
    training_algorithm = TRAINING_SETTINGS['training_algorithm']

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

    if algo_option['name'] == 'decision-tree':
        class_weight = algo_option['class_weight'] if algo_option.has_key('class_weight') else None
        random_state = algo_option['random_state'] if algo_option.has_key('random_state') else None
        criterion = algo_option['criterion'] if algo_option.has_key('criterion') else 'gini'

        return DecisionTreeClassifier(
            class_weight= class_weight,
            random_state=random_state,
            criterion=criterion)

    if algo_option['name'] == "random-forest":
        random_state = algo_option['random_state'] if algo_option.has_key('random_state') else None
        class_weight = algo_option['class_weight'] if algo_option.has_key('class_weight') else {0:1,1:1,2:1}
        return RandomForestClassifier(random_state=random_state, class_weight=class_weight)

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

    logo = StratifiedShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    fold_accuracy_scores = np.zeros(0)
    fold_f1_macro_scores = np.zeros(0)
    fold_f1_weighted_scores = np.zeros(0)
    fold_recall_scores = []
    fold_precision_scores = []

    # 5 folds corresponding to 5 events

    for train_index, test_index in logo.split(X_subset, y_subset):

        # Split train and test from folds
        X_train, X_test = get_array(train_index, X_subset), get_array(test_index, X_subset)
        y_train, y_test = get_array(train_index, y_subset), get_array(test_index, y_subset)

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

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        matrix = confusion_matrix(np.asarray(y_test), y_pred)
        # false_false_rate = 1.0* matrix[0][1] / sum(matrix[0])  # could be high
        # false_true_rate = 1.0* matrix[1][0] / sum(matrix[:, 0])  # must be low
        current_fold_accuracy = f1_score(np.asarray(y_test), y_pred, average='micro')
        current_fold_macro_f1 = f1_score(np.asarray(y_test), y_pred, average='macro')
        current_fold_weighted_f1 = f1_score(np.asarray(y_test), y_pred, average='weighted')
        current_recall = recall_score(np.asarray(y_test), y_pred, average=None)
        current_precision = precision_score(np.asarray(y_test), y_pred, average=None)

        # print "Micro f1-score (Accuracy):\t\t\t", current_fold_accuracy
        # print "Macro f1-score:\t\t\t", current_fold_macro_f1
        # print "Weighted f1-score:\t\t\t", current_fold_weighted_f1
        # print "Rate false of false label:\t\t\t", false_false_rate
        # print "Rate false of true label:\t\t\t", false_true_rate

        fold_accuracy_scores = np.append(fold_accuracy_scores,current_fold_accuracy)
        fold_f1_macro_scores = np.append(fold_f1_macro_scores, current_fold_macro_f1)
        fold_f1_weighted_scores = np.append(fold_f1_weighted_scores, current_fold_weighted_f1)
        fold_recall_scores.append(current_recall)
        fold_precision_scores.append(current_precision)
        # print current_recall
        # print current_precision
        # print confusion_matrix(np.asarray(y_test), y_pred)
        # tmp = []
        # for (index,x) in enumerate(model.feature_importances_):
        #     if x!=0:
        #         tmp.append((x,index))
        # print sorted(tmp, reverse=True)
        # raw_input()



    # print "Accuracy:\t\t", fold_accuracy_scores, '\t\t', fold_accuracy_scores.mean()
    # print "F1-macro:\t\t", fold_f1_macro_scores, '\t\t', fold_f1_macro_scores.mean()
    # print "F1-weighted:\t", fold_f1_weighted_scores, '\t\t', fold_f1_weighted_scores.mean()

    print "Accuracy:\t\t", fold_accuracy_scores.mean()
    print "F1-macro:\t\t",  fold_f1_macro_scores.mean()
    print "F1-weighted:\t", fold_f1_weighted_scores.mean()
    print "Recall: \t\t", np.asarray(fold_recall_scores).mean(axis=0)
    print "Precision: \t\t", np.asarray(fold_precision_scores).mean(axis=0)


    # TRAIN AND SAVE A MODEL FOR TESTING ON SEMEVAL TEST SET

    X_train = X_subset
    y_train = y_subset

    # Init a classifer
    model = init_model(algo_option)

    # Init an optional scaler
    scaler = None
    if scale_option:
        scaler = init_scaler(scale_option)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

    # Init an optional balancing model
    balancer = None
    if balancing_option:
        balancer = init_balancing_model(balancing_option)
        X_train, y_train = balancer.fit_sample(X_train, y_train)

    # Init an optional reduce dimenstion model
    reducer = None
    if reduce_dimension_option:
        reducer = init_reduce_dimension_model(reduce_dimension_option)
        reducer.fit(X_train, y_train)
        X_train = reducer.transform(X_train)

    # Fit prerocessed data to classifer model
    model.fit(X_train, y_train)

    # Save model
    pickle.dump(model, open(os.path.join(MODELS_ROOT,'classifier.model'),"wb"))

    if os.path.exists(os.path.join(MODELS_ROOT, 'scaler.model')):
        os.remove(os.path.join(MODELS_ROOT, 'scaler.model'))
    if scaler != None:
        pickle.dump(scaler, open(os.path.join(MODELS_ROOT, 'scaler.model'), "wb"))

    if os.path.exists(os.path.join(MODELS_ROOT, 'balancer.model')):
        os.remove(os.path.join(MODELS_ROOT, 'balancer.model'))
    if balancer != None:
        pickle.dump(balancer, open(os.path.join(MODELS_ROOT, 'balancer.model'), "wb"))

    if os.path.exists(os.path.join(MODELS_ROOT, 'reducer.model')):
        os.remove(os.path.join(MODELS_ROOT, 'reducer.model'))
    if reducer != None:
        pickle.dump(reducer, open(os.path.join(MODELS_ROOT, 'reducer.model'), "wb"))

    training_settings = {
        'features_subset': feature_option,
        'balancing_class_algorithm': balancing_option,
        'scale_option': scale_option,
        'reduce_dimension_algorithm': reduce_dimension_option,
        'training_algorithm': algo_option
    }

    pickle.dump(training_settings, open(os.path.join(MODELS_ROOT,'settings.model'),"wb"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
