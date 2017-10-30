from settings import TRAINING_OPTIONS, MODELS_ROOT
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from utils import read_training_processed_data
from sklearn.model_selection import LeaveOneGroupOut,  cross_val_score, cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import logging
import os


def instance_based( k):
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto')
    return model


def svm(X, y):
    return True


def main():
    # Read processed file
    X, y, groups = read_training_processed_data()

    # Train
    for option in TRAINING_OPTIONS[0:1]:
        if option == 'instance-based':
            for k in range(5, 50, 5):
                lda = LinearDiscriminantAnalysis()
                lda.fit(X, y)
                X_r2 = lda.transform(X)
                model = instance_based(k)
                print len(X[0])
                pickle.dump(lda, open(os.path.join(MODELS_ROOT, option + '-lda'+ '.model'), "wb"))
                pickle.dump(model, open(os.path.join(MODELS_ROOT, option + '.model'), "wb"))
                print k, cross_val_score(model, X_r2, y, groups = groups, cv = LeaveOneGroupOut()).mean()



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
