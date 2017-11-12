#!/usr/bin/python
import nltk
import itertools
from utils import preprocess_tweet

# Just prepare once for 4 cases for using after
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



def get_ngram_postag_vector(tweet, n):
    """
    Return the ngram POStagging vector of the tweet
    :param tweet: A nonpreprocessed tweet
    :param n: the number of gram in range [1,4]
    :return: Vector of ngram tagging using Universal tagging
    """
    #prepare the tag
    ngram_tag = prepare_tag(n)
    #preprocess tweet, remove emoticons, hashtags, metions
    tweet=preprocess_tweet(tweet)

    #tokenize tweet
    token = nltk.word_tokenize(tweet)
    tagged_token = nltk.pos_tag(token, tagset="universal")

    #create the vector size of ngram_tag
    pos_vector = [0] * len(ngram_tag)

    #check tag and return vector
    for i in range(0, (len(tagged_token) - n + 1)):
        str_list = []
        for j in range(0, n):
            str_list.append("'" + tagged_token[i+j][1] + "'")
        str1=", ".join(str_list)
        str="("+str1+")"
        pos_vector[(ngram_tag.index(str))] = 1

    return pos_vector
