from src.data.constants import DATASET_EVENTS, DATASET_NAME, TESTSET_NAME
from settings import *
from src.models import feature_bitmask
from copy import deepcopy


def read_training_processed_data():
    """
    Read vectors of features and labels from processed/ folder
    :return: Vector X,y, group corresponding feature vector, label vector and group vector
    """
    X = []
    y = []
    group = []

    processed_folder_path = os.path.join(DATA_PROCESSED_ROOT, DATASET_NAME)
    for group_idx, event in enumerate(DATASET_EVENTS):
        event_folder_path = os.path.join(processed_folder_path, event)
        train_file = open(os.path.join(event_folder_path, "train.txt"),"r")
        feature_vectors = train_file.read().splitlines()
        label_file = open(os.path.join(event_folder_path, "train_label.txt"),"r")
        labels = label_file.read().splitlines()
        for idx, label in enumerate(labels):
            y.append(int(label))
            X.append(map(float,feature_vectors[idx].split('\t')))
            group.append(group_idx)

    return X,y, group


def read_testing_processed_data():
    """
    Read vectors of features and labels from processed/ folder
    :return: Vector X,y, group corresponding feature vector, label vector
    """
    X = []
    y = []

    processed_folder_path = os.path.join(DATA_PROCESSED_ROOT, TESTSET_NAME)

    train_file = open(os.path.join(processed_folder_path, "test.txt"),"r")
    feature_vectors = train_file.read().splitlines()
    label_file = open(os.path.join(processed_folder_path, "test_label.txt"),"r")
    labels = label_file.read().splitlines()
    for idx, label in enumerate(labels):
        y.append(int(label))
        X.append(map(float,feature_vectors[idx].split('\t')))

    return X, y


def get_subset_features(X, feature_option):
    """
        Get a subset of features from predefined settings
    :param feature_option: a string as a name of settings
            X: original features
    :return: subset of result features
    """
    X_new = []
    for instance in X:
        instance_new = []
        for feature_index in FEATURE_OPTIONS[feature_option]:
            start, end = feature_bitmask[FEATURE_LIST[feature_index]]
            instance_new += instance[start : end]
        X_new.append(instance_new)

    return X_new


def get_array(idx_array, x):
    return [x[i] for i in idx_array]
