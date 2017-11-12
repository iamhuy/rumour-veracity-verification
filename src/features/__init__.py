import os
from settings import MODELS_ROOT, DATA_EXTERNAL_ROOT
from constants import brown_cluster_dict_filename
from utils import read_brown_cluster_file
import pickle
from src.lib.ark_twokenize_py import twokenize
import re
import itertools

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