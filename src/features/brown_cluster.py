# -*- coding: utf-8 -*-
from src.lib.ark_twokenize_py import twokenize


def brown_cluster(tweet_text):
    """
        Get a distribution of brown cluster of a tweet
    :param tweet_text: Tweet content as a string
    :return: A vector of size-1000, each element 1 represents an existence of a cluster in tweet, 0 otherwise
    """

    list_token = twokenize.tokenizeRawTweetText(tweet_text)
    clusters = [0 for _ in range(1000)]
    for token in list_token:
        
