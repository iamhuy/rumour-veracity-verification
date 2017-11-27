from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import logging
import os
from settings import MODELS_ROOT
from utils import read_testing_processed_data
from semeval_scorer import sem_eval_score
from collections import Counter
from sklearn.metrics import confusion_matrix, f1_score
from utils import get_subset_features
import numpy as np

def main():
    X_test, y_test = read_testing_processed_data()

    reducer = None
    scaler = None
    balancer = None

    settings = pickle.load(open(os.path.join(MODELS_ROOT, 'settings.model'), "rb"))
    X_test = get_subset_features(X_test, feature_option=settings['features_subset'])
    print len(X_test[0])

    classifier = pickle.load(open(os.path.join(MODELS_ROOT, 'classifier.model'), "rb"))

    if os.path.exists(os.path.join(MODELS_ROOT, 'scaler.model')):
        scaler = pickle.load(open(os.path.join(MODELS_ROOT, 'scaler.model'), "rb"))
        X_test = scaler.transform(X_test)

    # if os.path.exists(os.path.join(MODELS_ROOT, 'balancer.model')):
    #     balancer = pickle.load(open(os.path.join(MODELS_ROOT, 'balancer.model'), "rb"))
    #     X_test = balancer.transform(X_test)

    if os.path.exists(os.path.join(MODELS_ROOT, 'reducer.model')):
        reducer = pickle.load(open(os.path.join(MODELS_ROOT, 'reducer.model'), "rb"))
        X_test = reducer.transform(X_test)

    print len(X_test[0])
    y_pred_prob = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test).tolist()
    y_actual = [(label, y_pred_prob[idx][label]) for idx, label in enumerate(y_pred)]

    matrix = confusion_matrix(np.asarray(y_test), y_pred)
    false_recall = 1.0 * matrix[1][1] / sum(matrix[0])  # must be high
    true_recall = 1.0 * matrix[0][0] / sum(matrix[1])  # can be low
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