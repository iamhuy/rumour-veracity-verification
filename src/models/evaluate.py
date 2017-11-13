from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import logging
import os
from settings import TRAINING_OPTIONS, MODELS_ROOT
from utils import read_testing_processed_data
from semeval_scorer import sem_eval_score


def main():
    X, y_gold = read_testing_processed_data()
    for option in TRAINING_OPTIONS[0:1]:
        lda = None
        classifier = None
        if option == 'instance-based':
            lda = pickle.load(open(os.path.join(MODELS_ROOT, option + '-lda'+ '.model'), "rb"))
            classifier = pickle.load(open(os.path.join(MODELS_ROOT, option + '.model'), "rb"))
        X_new = lda.transform(X)
        y_actual_prob = classifier.predict_proba(X_new)
        y_actual_label = classifier.predict(X_new).tolist()
        y_actual = [(label, y_actual_prob[idx][label])for idx, label in enumerate(y_actual_label)]
        print y_actual_label
        print y_gold
        sem_eval_score(y_actual, y_gold)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()