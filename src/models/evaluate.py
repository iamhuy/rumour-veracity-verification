from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import logging
import os
from settings import TRAINING_OPTIONS, MODELS_ROOT
from utils import read_testing_processed_data


def main():
    X, y_gold = read_testing_processed_data()

    for option in TRAINING_OPTIONS[0:1]:
        lda = None
        classifier = None
        if option == 'instance-based':
            lda = pickle.load(open(os.path.join(MODELS_ROOT, option + '-lda'+ '.model'), "rb"))
            classifier = pickle.load(open(os.path.join(MODELS_ROOT, option + '.model'), "rb"))
        X_new = lda.transform(X)
        y_actual = classifier.predict(X_new)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()