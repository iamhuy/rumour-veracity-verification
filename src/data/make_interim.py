# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv
from constants import *
import json
from utils import json_from_file, merge_json
import shutil
from settings import *

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        interim data to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    # Init absolute path of folders
    raw_input_folder_path = os.path.join(DATA_RAW_ROOT, DATASET_NAME, RAW_INPUT_FOLDER)
    raw_output_folder_path = os.path.join(DATA_RAW_ROOT, DATASET_NAME, RAW_OUTPUT_FOLDER)
    interim_folder_path = os.path.join(DATA_INTERIM_ROOT, DATASET_NAME)

    # Read veracities from both test and dev files
    veracity_labels = merge_json(
        json_from_file(os.path.join(raw_output_folder_path, VERACITY_LABEL_FILE[0])),
        json_from_file(os.path.join(raw_output_folder_path, VERACITY_LABEL_FILE[1])))

    # Read stances from both test and dev files
    stance_labels = merge_json(
        json_from_file(os.path.join(raw_output_folder_path, STANCE_LABEL_FILE[0])),
        json_from_file(os.path.join(raw_output_folder_path, STANCE_LABEL_FILE[1])))

    # If interim data existed, delete and create a new one
    if os.path.exists(interim_folder_path):
        shutil.rmtree(interim_folder_path)
    os.makedirs(interim_folder_path)

    for event_name in DATASET_EVENTS:
        interim_event_folder_path = os.path.join(interim_folder_path, event_name)
        os.makedirs(interim_event_folder_path)
        event_folder_path = os.path.join(raw_input_folder_path, event_name)
        list_tweet_ids = [name for name in os.listdir(event_folder_path) if os.path.isdir(os.path.join(event_folder_path,name))]

        for index, id in enumerate(list_tweet_ids):

            # thread conversation folder in raw
            source_tweet_folder_path =  os.path.join(event_folder_path, id)

            # read source tweet
            source_tweet_file = open(os.path.join(source_tweet_folder_path,'source-tweet', id + '.json'), 'r')
            source_tweet_content = source_tweet_file.read()
            source_tweet_file.close()
            source_tweet = json.loads(source_tweet_content)
            source_tweet_replies = []

            # read replies
            replies_folder_path = os.path.join(source_tweet_folder_path, 'replies')
            list_reply_ids = [name for name in os.listdir(replies_folder_path) if os.path.isfile(os.path.join(replies_folder_path, name))]
            for reply_id in list_reply_ids:
                reply_file = open(os.path.join(replies_folder_path, reply_id), "r")
                reply_content = reply_file.read()
                reply_file.close()
                reply = json.loads(reply_content)
                reply['stance'] = stance_labels[reply['id_str']]
                source_tweet_replies.append(reply)

            source_tweet['replies'] = source_tweet_replies

            # read structure
            structure_file = open(os.path.join(source_tweet_folder_path,'structure.json'), "r")
            structure_content = structure_file.read()
            structure_file.close()
            structure = json.loads(structure_content)
            source_tweet['structure'] = structure

            source_tweet['veracity'] = veracity_labels[source_tweet['id_str']]

            source_tweet['stance'] = stance_labels[source_tweet['id_str']]

            # create tweet file in interim to write
            interim_tweet_file = open(os.path.join(interim_event_folder_path, str(index) + '.json'), "w")

            # write tweet to interim
            interim_tweet_file.write(json.dumps(source_tweet, indent = 4))
            interim_tweet_file.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
