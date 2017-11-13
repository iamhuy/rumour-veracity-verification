from settings import MODELS_ROOT, DATA_EXTERNAL_ROOT
from constants import brown_cluster_dict_filename
from utils import read_brown_cluster_file
from src.lib.ark_twokenize_py import twokenize
import re
import gensim
import os
import pickle
import nltk
import itertools
from nltk.tag.stanford import StanfordNERTagger
import time


# Read brown cluster from dict or from text file

brown_cluster_dict_filepath = os.path.join(MODELS_ROOT, brown_cluster_dict_filename)
brown_cluster_text_filepath = os.path.join(DATA_EXTERNAL_ROOT, '50mpaths2.txt')
brown_cluster_dict = None


if os.path.exists(brown_cluster_dict_filepath):
    brown_cluster_dict = pickle.load(open(brown_cluster_dict_filepath, "rb"))
else:
    brown_cluster_text_file = open(brown_cluster_text_filepath, "r")
    brown_cluster_dict = read_brown_cluster_file(brown_cluster_text_file)
    pickle.dump(brown_cluster_dict, open(brown_cluster_dict_filepath, "wb"))

mention_regex = re.compile('^' + twokenize.AtMention + '$')
url_regex = re.compile('^' + twokenize.url+ '$')
url2_regex = re.compile(r"^((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?$")



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



# Load Google's pre-trained Word2Vec model.

t1 = time.time()
print "Starting loading word2vec model !"
google_word2Vec_path=os.path.join(DATA_EXTERNAL_ROOT, 'GoogleNews-vectors-negative300.bin')
google_word2Vec_model = gensim.models.KeyedVectors.load_word2vec_format(google_word2Vec_path, binary=True)
print "Finish loading word2vec model !"
print time.time() - t1

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
nltk.download('stopwords')



#Prepare the tag-set

def prepare_tag(n):
    """
    Prepare the combination of the tagset
    :param n: the number of gram
    :return: the tag set relating to n
    """
    tag_set = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    ngram_tag=[]
    if n == 1:
        for i in tag_set:
            ngram_tag.append("('"+i+"')")
    elif n == 2:
        for i in itertools.product(tag_set, tag_set):
            ngram_tag.append(str(i))
    elif n == 3:
        for i in itertools.product(tag_set, tag_set, tag_set):
            ngram_tag.append(str(i))
    elif n == 4:
        for i in itertools.product(tag_set, tag_set, tag_set, tag_set):
            ngram_tag.append(str(i))
    return ngram_tag

monogram_tagset = prepare_tag(1)
bigram_tagset = prepare_tag(2)
trigram_tagset = prepare_tag(3)
fourgram_tagset = prepare_tag(4)


# Read the Stanford NER Parser

stanford_ner_classifier = os.path.join(DATA_EXTERNAL_ROOT, 'stanford-ner-2017-06-09', 'classifiers', 'english.all.3class.distsim.crf.ser.gz')
stanford_ner_jar = os.path.join(DATA_EXTERNAL_ROOT, 'stanford-ner-2017-06-09', 'stanford-ner.jar')
stanford_ner = StanfordNERTagger(stanford_ner_classifier, stanford_ner_jar)

