import gensim
import os
import pickle
from src.features import google_word2Vec_model
from settings import DATA_EXTERNAL_ROOT

# Load Google's pre-trained Word2Vec model.

google_word2Vec_path=os.path.join(DATA_EXTERNAL_ROOT, 'GoogleNews-vectors-negative300.bin')
google_word2Vec_model = gensim.models.KeyedVectors.load_word2vec_format(google_word2Vec_path,
                                                                        binary=True, limit=1000000)



# Load the wordList of surprise, doubt, nodoubt

def get_wordlist(filename):
    """
    Read the list from the file
    :param filename: the name of file of words
    :return: list of synonyms
    """
    with open(filename, 'rb') as f:
        wordList = pickle.load(f)
    return wordList

surprisePath = os.path.join(DATA_EXTERNAL_ROOT, 'surprise_list_file')
doubtPath = os.path.join(DATA_EXTERNAL_ROOT, 'doubt_list_file')
noDoubtPath = os.path.join(DATA_EXTERNAL_ROOT, 'nodoubt_list_file')

surpriseList = get_wordlist(surprisePath)
doubtList = get_wordlist(doubtPath)
noDoubtList = get_wordlist(noDoubtPath)