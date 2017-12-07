from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import logging
import os
from settings import MODELS_ROOT
from utils import *
from semeval_scorer import sem_eval_score
from collections import Counter
from sklearn.metrics import confusion_matrix, f1_score
from utils import get_subset_features
import numpy as np
from copy import deepcopy

def main():
    X_test, y_test = read_testing_processed_data()

    reducer = None
    scaler = None
    balancer = None
    feature_selector = None

    settings = pickle.load(open(os.path.join(MODELS_ROOT, 'settings.model'), "rb"))
    # X_test = get_subset_features(X_test, feature_option=settings['features_subset'])

    if settings['scale_option'] != None:
        scaler = pickle.load(open(os.path.join(MODELS_ROOT, 'scaler.model'), "rb"))
        X_test = scaler.transform(X_test)

    if settings['feature_selection_algorithm'] != None:
        feature_selector = pickle.load(open(os.path.join(MODELS_ROOT, 'feature_selector.model'), "rb"))
        X_test = feature_selector.transform(X_test)

    if settings['reduce_dimension_algorithm'] != None:
        reducer = pickle.load(open(os.path.join(MODELS_ROOT, 'reducer.model'), "rb"))
        X_test = reducer.transform(X_test)

    classifier = pickle.load(open(os.path.join(MODELS_ROOT, 'classifier.model'), "rb"))

    print len(X_test[0])
    y_pred_prob = classifier.predict_proba(X_test)
    y_pred = predict_with_false_priority(y_pred_prob, false_priority=1.0)
    # y_pred = classifier.predict(X_test).tolist()
    y_actual = [(label, y_pred_prob[idx][label]) for idx, label in enumerate(y_pred)]

    matrix = confusion_matrix(np.asarray(y_test), y_pred)
    false_recall = 1.0 * matrix[1][1] / sum(matrix[1])  # must be high
    true_recall = 1.0 * matrix[0][0] / sum(matrix[0])  # can be low
    unverified_recall = 1.0 * matrix[2][2] / sum(matrix[2]) # must be high
    accuracy = f1_score(np.asarray(y_test), y_pred, average='micro')
    macro_f1 = f1_score(np.asarray(y_test), y_pred, average='macro')
    weighted_f1 = f1_score(np.asarray(y_test), y_pred, average='weighted')

    print matrix
    print "Micro f1-score (Accuracy):\t", accuracy
    print "Macro f1-score:\t\t\t\t", macro_f1
    print "Weighted f1-score:\t\t\t", weighted_f1
    print "False Recall:\t\t\t\t", false_recall
    print "Unverified Recall:\t\t\t", unverified_recall
    print "True Recall:\t\t\t\t", true_recall
    sem_eval_score(y_actual, y_test)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()