from settings import TRAINING_OPTIONS, MODELS_ROOT
from sklearn.neighbors import KNeighborsClassifier
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
    X = [[0,0,1],[1,1,0],[0,0,0]]
    y = [0,1,2]
    # Get Features


    # Train
    for option in TRAINING_OPTIONS:
        if option == 'instance-based':
            model = instance_based(X, y)
            pickle.dump(model, open(os.path.join(MODELS_ROOT, option + '.model'), "wb"))
        # if option == 'svm':
        # if option == 'j48':



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
