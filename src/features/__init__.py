import os
from settings import MODELS_ROOT, DATA_EXTERNAL_ROOT
from constants import brown_cluster_dict_filename
from utils import read_brown_cluster_file
import pickle
from src.lib.ark_twokenize_py import twokenize
import re

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