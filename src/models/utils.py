from src.data.constants import DATASET_EVENTS, DATASET_NAME
from settings import *

def read_processed_data():
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
<<<<<<< Updated upstream
            X.append(map(int,feature_vectors[idx].split('\t')))
=======
            X.append(map(float,feature_vectors[idx].split('\t')))
>>>>>>> Stashed changes
            group.append(group_idx)

    return X,y, group


