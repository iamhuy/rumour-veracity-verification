from macpath import norm_error
from sklearn.naive_bayes import GaussianNB
from settings import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import read_training_processed_data, get_feature_name
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
from utils import *
import shutil
from sklearn.feature_selection import *
from sklearn.linear_model import *


def main():
    X, y, groups = read_training_processed_data()

    # X_add, y_add = read_testing_processed_data()
    #
    # X = X + X_add
    # y = y + y_add

    print len(X)
    np.set_printoptions(precision=4)

    features_subset = TRAINING_SETTINGS['features_subset']
    balancing_class_algorithm = TRAINING_SETTINGS['balancing_class_algorithm']
    scale_option = TRAINING_SETTINGS['scale_option']
    reduce_dimension_algorithm = TRAINING_SETTINGS['reduce_dimension_algorithm']
    training_algorithm = TRAINING_SETTINGS['training_algorithm']
    feature_selection_algorithm = TRAINING_SETTINGS['feature_selection_algorithm']

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
        feature_selection_option = feature_selection_algorithm
    )


def init_model(algo_option):
    if algo_option['name'] == 'knn':
        k = algo_option['k'] if algo_option.has_key('k') else 5
        return KNeighborsClassifier(n_neighbors = k)

    if algo_option['name'] == 'decision-tree':
        class_weight = algo_option['class_weight'] if algo_option.has_key('class_weight') else None
        random_state = algo_option['random_state'] if algo_option.has_key('random_state') else None
        criterion = algo_option['criterion'] if algo_option.has_key('criterion') else 'gini'

        return DecisionTreeClassifier(class_weight= class_weight, random_state=random_state, criterion=criterion)

    if algo_option['name'] == "random-forest":
        random_state = algo_option['random_state'] if algo_option.has_key('random_state') else None
        class_weight = algo_option['class_weight'] if algo_option.has_key('class_weight') else {0:1,1:1,2:1}
        return RandomForestClassifier(random_state=random_state, class_weight=class_weight)

    if algo_option['name'] == "gradient-boosting":
        random_state = algo_option['random_state'] if algo_option.has_key('random_state') else None
        n_estimators = algo_option['n_estimators'] if algo_option.has_key('n_estimators') else 100
        learning_rate = algo_option['learning_rate'] if algo_option.has_key('learning_rate') else 0.1
        verbose = algo_option['verbose'] if algo_option.has_key('verbose') else False
        return GradientBoostingClassifier(random_state=random_state, n_estimators=n_estimators, learning_rate=learning_rate, verbose=verbose)

    if algo_option['name'] == "extra-trees":
        random_state = algo_option['random_state'] if algo_option.has_key('random_state') else None
        n_estimators = algo_option['n_estimators'] if algo_option.has_key('n_estimators') else 100
        class_weight = algo_option['class_weight'] if algo_option.has_key('class_weight') else None
        bootstrap = algo_option['bootstrap'] if algo_option.has_key('bootstrap') else False
        return ExtraTreesClassifier(random_state=random_state, n_estimators=n_estimators, class_weight=class_weight, bootstrap=bootstrap)

    return None


def init_balancing_model(balancing_option):
    if balancing_option['name'] == 'SMOTE':
        random_state = balancing_option['random_state'] if balancing_option.has_key('random_state') else None
        return SMOTE(k_neighbors = balancing_option['k'], random_state=random_state)

    return None


def init_scaler(scale_option):
    if scale_option['name'] == 'MaxAbs':
        return preprocessing.MaxAbsScaler()

    if scale_option['name'] == 'MinMax':
        return preprocessing.MinMaxScaler()

    if scale_option['name'] == 'Robust':
        return preprocessing.RobustScaler()

    if scale_option['name'] == 'Standard':
        return preprocessing.StandardScaler()

    return None


def init_reduce_dimension_model(reduce_dimension_option):
    if reduce_dimension_option['name'] == 'PCA':
        random_state = reduce_dimension_option['random_state'] if reduce_dimension_option.has_key('random_state') else None
        return PCA(n_components=reduce_dimension_option['n_components'], random_state=random_state)

    return None


def init_feature_selection_model(feature_selection_option):
    if feature_selection_option['name'] == 'variance-threshold':
        threshold = feature_selection_option['threshold'] if feature_selection_option.has_key('threshold') else 0
        return VarianceThreshold(threshold=threshold)


    if feature_selection_option['name'] == 'k-best':
        k = feature_selection_option['k'] if feature_selection_option.has_key('k') else 10
        score_func = feature_selection_option['score_func'] if feature_selection_option.has_key('score_func') else f_classif
        return SelectKBest(score_func=score_func, k=k)

    if feature_selection_option['name'] == 'fpr':
        score_func = feature_selection_option['score_func'] if feature_selection_option.has_key('score_func') else f_classif
        alpha = feature_selection_option['alpha'] if feature_selection_option.has_key('alpha') else 0.05
        return SelectFpr(score_func=score_func, alpha=alpha)

    if feature_selection_option['name'] == 'fdr':
        score_func = feature_selection_option['score_func'] if feature_selection_option.has_key('score_func') else f_classif
        alpha = feature_selection_option['alpha'] if feature_selection_option.has_key('alpha') else 0.05
        return SelectFdr(score_func=score_func, alpha=alpha)

    if feature_selection_option['name'] == 'fwe':
        score_func = feature_selection_option['score_func'] if feature_selection_option.has_key('score_func') else f_classif
        alpha = feature_selection_option['alpha'] if feature_selection_option.has_key('alpha') else 0.05
        return SelectFwe(score_func=score_func, alpha=alpha)

    if feature_selection_option['name'] == 'generic':
        score_func = feature_selection_option['score_func'] if feature_selection_option.has_key('score_func') else f_classif
        return GenericUnivariateSelect(score_func=score_func)

    if feature_selection_option['name'] == 'from-model':
        estimator = \
            feature_selection_option['model'] if feature_selection_option.has_key('model') else LassoCV()
        threshold = feature_selection_option['threshold'] if feature_selection_option.has_key('threshold') else 'mean'
        prefit =  feature_selection_option['prefit'] if feature_selection_option.has_key('prefit') else False
        return SelectFromModel(estimator=estimator, threshold=threshold, prefit=prefit)

    return None


def train(X ,y, groups, algo_option, feature_option, balancing_option, scale_option, reduce_dimension_option, feature_selection_option):

    # Read processed file
    # X_subset = get_subset_features(X, feature_option)
    # y_subset = deepcopy(y)

    X_subset = X
    y_subset = y

    feature_selector = None
    if feature_selection_option != None:
        feature_selector = init_feature_selection_model(feature_selection_option)
        feature_selector.fit(X_subset, y_subset)
        X_subset = feature_selector.transform(X_subset)

    # print sorted(feature_selector.variances_, reverse=True)
    print feature_selector.get_params
    print feature_selector.get_support(indices=True)
    print len(feature_selector.get_support(indices=True))
    print Counter([get_feature_name(index) for index in feature_selector.get_support(indices=True)]).items()

    logo = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    fold_accuracy_scores = np.zeros(0)
    fold_f1_macro_scores = np.zeros(0)
    fold_f1_weighted_scores = np.zeros(0)
    fold_recall_scores = []
    fold_precision_scores = []

    # 5 folds corresponding to 5 events
    i = 0

    for train_index, test_index in logo.split(X_subset, y_subset):
        print i
        i+=1
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
        y_pred_prob = model.predict_proba(X_test)
        y_pred = predict_with_false_priority(y_pred_prob, false_priority=1.0)
        # y_pred = model.predict(X_test)

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

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    feature_ranking = [(i,get_feature_name(i)) for i in sorted_idx]
    print feature_ranking


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

    if os.path.exists(os.path.join(MODELS_ROOT, 'feature_selector.model')):
        os.remove(os.path.join(MODELS_ROOT, 'feature_selector.model'))
    if feature_selector != None:
        pickle.dump(feature_selector, open(os.path.join(MODELS_ROOT, 'feature_selector.model'), "wb"))

    training_settings = {
        'features_subset': feature_option,
        'balancing_class_algorithm': balancing_option,
        'scale_option': scale_option,
        'reduce_dimension_algorithm': reduce_dimension_option,
        'training_algorithm': algo_option,
        'feature_selection_algorithm': feature_selection_option
    }

    pickle.dump(training_settings, open(os.path.join(MODELS_ROOT,'settings.model'),"wb"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
