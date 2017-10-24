from settings import TRAINING_OPTIONS, MODELS_ROOT
from sklearn.neighbors import KNeighborsClassifier
from utils import read_processed_data
from sklearn.model_selection import LeaveOneGroupOut,  cross_val_score
import pickle
import logging
import os


def instance_based(X, y):
    model = KNeighborsClassifier()
    model.fit(X,y)
    return model


def svm(X, y):
    return True


def main():
    # Read processed file
    X, y, groups = read_processed_data()

    # Train
    for option in TRAINING_OPTIONS:
        if option == 'instance-based':
            model = instance_based(X, y)
            pickle.dump(model, open(os.path.join(MODELS_ROOT, option + '.model'), "wb"))
        # if option == 'svm':
        # if option == 'j48':

        print cross_val_score(model, X, y, groups = groups, cv = LeaveOneGroupOut())



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
