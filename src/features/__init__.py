import os
from settings import MODELS_ROOT, DATA_EXTERNAL_ROOT
#from constants import brown_cluster_dict_filename
#from utils import read_brown_cluster_file
import pickle
from src.lib.ark_twokenize_py import twokenize
import re

# Read the list of the bad words, acronyms

def readList(filename):
    """
    Read the saved file list containing words
    :param filename: the name of the file
    :return: the list of words
    """
    wordList=[]
    with open(filename, 'rb') as fp:
        wordList = pickle.load(fp)
        #print wordList
    return wordList

google_bad_words_path=os.path.join(DATA_EXTERNAL_ROOT,'google_bad_words_list')
noswearing_bad_words_path=os.path.join(DATA_EXTERNAL_ROOT,'noswearing_bad_words_list')
netlingo_acronyms_path=os.path.join(DATA_EXTERNAL_ROOT,'netlingo_acronyms_list')

google_bad_words_list=readList(google_bad_words_path)
noswearing_bad_words_list=readList(noswearing_bad_words_path)
netlingo_acronyms_list=readList(netlingo_acronyms_path)

# Read the model for Stanford NLP

