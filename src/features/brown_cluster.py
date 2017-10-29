# -*- coding: utf-8 -*-
from src.lib.ark_twokenize_py import twokenize
from . import url_regex, url2_regex, mention_regex, brown_cluster_dict
import re

def brown_cluster(tweet_text):
    """
        Get a distribution of brown cluster of a tweet and check if it contains URL or not
    :param tweet_text: Tweet content as a string
    :return: A vector of size-1000, each element 1 represents an existence of a cluster in tweet, 0 otherwise
    """

    has_url = False
    list_token = twokenize.tokenizeRawTweetText(tweet_text.lower())
    clusters = [0 for _ in range(1000)]
    for token in list_token:
        word = token
        # match url
        matchObj = url_regex.match(token)
        matchGroups = url2_regex.match(token)
        if matchObj != None and matchGroups != None:
            has_url = True
            word = "<URL-{0}>".format(matchGroups.group(3))

        # match mention
        matchObj = mention_regex.match(token)
        if matchObj != None:
            word = "<@MENTION>"

        if brown_cluster_dict.has_key(word):
            clusters[brown_cluster_dict[word]] = 1

    return clusters, has_url