# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv
from constants import *
from utils import json_from_file
from src.features.build_features import collect_feature
import shutil
from settings import *

def prepare_processed_training_data():
    """
    Generate all features into processes/... folder from interim/...
    """
    logger = logging.getLogger(__name__)
    logger.info('Making processed training data set from interim data')

    # Init absolute path of folders
    processed_folder_path = os.path.join(DATA_PROCESSED_ROOT, DATASET_NAME)
    interim_folder_path = os.path.join(DATA_INTERIM_ROOT, DATASET_NAME)

    if os.path.exists(processed_folder_path):
        shutil.rmtree(processed_folder_path)
    os.makedirs(processed_folder_path)


    for event_name in DATASET_EVENTS:

        event_folder_path = os.path.join(interim_folder_path, event_name)
        list_tweet_ids = [name for name in os.listdir(event_folder_path) if
                          os.path.isfile(os.path.join(event_folder_path, name))]

        processed_event_folder_path = os.path.join(processed_folder_path, event_name)
        os.makedirs(processed_event_folder_path)

        train_processed_file = open(os.path.join(processed_event_folder_path, 'train.txt'),"w")
        train_processed_label_file = open(os.path.join(processed_event_folder_path, 'train_label.txt'), "w")

        tweet_count = len(list_tweet_ids)

        for index, id  in enumerate(list_tweet_ids):
            print event_name , '+', index
            source_tweet = json_from_file(os.path.join(event_folder_path, id))
            features = collect_feature(source_tweet)
            features_str = "\t".join([str(i) for i in features])
            train_processed_file.write(features_str)
            if index != tweet_count-1 :
                train_processed_file.write('\n')
            train_processed_label_file.write(str(VERACITY_LABELS_MAPPING[source_tweet['veracity']]))
            if index != tweet_count-1 :
                train_processed_label_file.write('\n')

        train_processed_file.close()
        train_processed_label_file.close()


def prepare_processed_testing_data():
    """
    Generate all features into processes/... folder from interim/...
    """
    logger = logging.getLogger(__name__)
    logger.info('Making processed testing data set from interim data')

    # Init absolute path of folders
    processed_folder_path = os.path.join(DATA_PROCESSED_ROOT, TESTSET_NAME)
    interim_folder_path = os.path.join(DATA_INTERIM_ROOT, TESTSET_NAME)

    if os.path.exists(processed_folder_path):
        shutil.rmtree(processed_folder_path)
    os.makedirs(processed_folder_path)

    list_tweet_ids = [name for name in os.listdir(interim_folder_path) if
                      os.path.isfile(os.path.join(interim_folder_path, name))]

    test_processed_file = open(os.path.join(processed_folder_path, 'test.txt'),"w")
    test_processed_label_file = open(os.path.join(processed_folder_path, 'test_label.txt'), "w")

    tweet_count = len(list_tweet_ids)

    for index, id  in enumerate(list_tweet_ids):
        source_tweet = json_from_file(os.path.join(interim_folder_path, id))
        features = collect_feature(source_tweet)
        features_str = "\t".join([str(i) for i in features])
        test_processed_file.write(features_str)
        if index != tweet_count-1 :
            test_processed_file.write('\n')
        test_processed_label_file.write(str(VERACITY_LABELS_MAPPING[source_tweet['veracity']]))
        if index != tweet_count-1 :
            test_processed_label_file.write('\n')

    test_processed_file.close()
    test_processed_label_file.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    prepare_processed_training_data()
    prepare_processed_testing_data()
