#!/usr/bin/python
from sklearn.feature_extraction.text import CountVectorizer
import os
import json
import preprocessor as p
import sys
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import nltk
import string
import itertools


def preprocess_tweet(tweet):
    cleaned_tweet = tweet.lower()  # lowercase the tweet
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)  # set options for the preprocessor
    cleaned_tweet = p.clean(cleaned_tweet.encode("ascii", "ignore"))

    return cleaned_tweet;


def get_bigram_postag_vector(tweet):
    """
    Return the bi-gram POStagging of the tweet
    :param tweet: A nonpreprocessed tweet
    :return: a universal postagging, bigram vector
    """
    tag_set = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    bigram_tag = []
    for i in itertools.product(tag_set, tag_set):
        bigram_tag.append(str(i))

    #preprocess tweet, remove emoticons, hashtags, metions
    tweet=preprocess_tweet(tweet)

    #tokenize tweet
    token = nltk.word_tokenize(tweet)
    tagged_token = nltk.pos_tag(token, tagset="universal")

    #create the vector size of bigram_tag
    pos_vector = [0] * len(bigram_tag)

    #check tag and return vector
    for i in range(0, (len(tagged_token) - 1)):
        pos_vector[
            (bigram_tag.index(str("('" + tagged_token[i][1] + "'" + ", " + "'" + tagged_token[i + 1][1] + "')")))] = 1

    return pos_vector


def get_trigram_postag_vector(tweet):
    """
    Return the bi-gram POStagging of the tweet
    :param tweet: A nonpreprocessed tweet
    :return: a universal postagging, bigram vector
    """
    tag_set = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    trigram_tag = []
    for i in itertools.product(tag_set, tag_set,tag_set):
        trigram_tag.append(str(i))

    #preprocess tweet, remove emoticons, hashtags, metions
    tweet=preprocess_tweet(tweet)

    #tokenize tweet
    token = nltk.word_tokenize(tweet)
    tagged_token = nltk.pos_tag(token, tagset="universal")

    #create the vector size of bigram_tag
    pos_vector = [0] * len(trigram_tag)

    #check tag and return vector
    for i in range(0, (len(tagged_token) - 2)):
        pos_vector[
            (trigram_tag.index(str("('" + tagged_token[i][1] + "'" + ", " + "'" + tagged_token[i + 1][1] + "'" + ", " + "'" + tagged_token[i + 2][1]+"')")))] = 1

    return pos_vector

