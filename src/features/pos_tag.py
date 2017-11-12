#!/usr/bin/python
import nltk
from utils import preprocess_tweet
from src.features import monogram_tagset, bigram_tagset, trigram_tagset, fourgram_tagset

def get_ngram_postag_vector(tweet, n):
    """
    Return the ngram POStagging vector of the tweet
    :param tweet: A nonpreprocessed tweet
    :param n: the number of gram in range [1,4]
    :return: Vector of ngram tagging using Universal tagging
    """
    #prepare the tag
    if n==1:
        ngram_tag=monogram_tagset
    elif n==2:
        ngram_tag=bigram_tagset
    elif n==3:
        ngram_tag=trigram_tagset
    elif n==4:
        ngram_tag=fourgram_tagset
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
