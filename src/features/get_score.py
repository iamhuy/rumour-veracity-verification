import re
import logging
import numpy as np
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from scipy.spatial.distance import cosine
from src.features import google_word2Vec_model, surpriseList, doubtList, noDoubtList

def remove_stopwords_and_tokenize(tweet):
    """
    Remove stopwords and tokenize the tweet
    :param tweet: raw tweet
    :return: the tokens of tweet with no stop word
    """
    stop_words = set(stopwords.words('english'))
    tweet = tweet.lower()
    # Remove stopwords
    tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    tweet_tokens = tweet_tokenizer.tokenize(tweet)
    no_stop_words = []
    for token in tweet_tokens:
        if not token in stop_words:
            no_stop_words.append(token)
            # remove words less than 2 letters

    no_stop_words = [re.sub(r'^\w\w?$', '', i) for i in no_stop_words]
    return no_stop_words


def preprocess_and_tokenize_tweet(tweet):
    """
    Preprocess tweet, remove url, emoji, mentions, hastags, stopwords
    :param tweet: raw tweet
    :return: tweet
    """
    cleaned_tweet = tweet.lower()  # lowercase the tweet
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)  # set options for the preprocessor
    cleaned_tweet = p.clean(cleaned_tweet.encode('ascii', 'ignore'))
    tweet_words = remove_stopwords_and_tokenize(cleaned_tweet)  # remove stopwords
    return tweet_words


def cumulative_vector_wordList(wordList):
    """
    Get cumulative word2Vec vector for a wordlist
    :param wordList: the list of words
    :return: the 300d if wordList is not empty, otherwise None. Also None when cannot find any word in the dictionary
    """
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    numOfWord = len(wordList)
    if numOfWord>0:
        cumulative_vector = np.zeros((300,), dtype=np.float)
        for word in wordList:
            try:
                cumulative_vector = np.add(cumulative_vector, google_word2Vec_model[word])
            except KeyError: #handle case that the word is not in the Oxford dictionary
                numOfWord -= 1
                continue
        if numOfWord == 0: #handle case that there is no word found in the dictionary
            return None
        else:
            return np.divide(cumulative_vector, numOfWord)
    else:
        return None


# Get cumulative vectors
surpriseVector = cumulative_vector_wordList(surpriseList)
doubtVector = cumulative_vector_wordList(doubtList)
noDoubtVector = cumulative_vector_wordList(noDoubtList)



def get_vectors(tweet):
    """
    Get vectors of cosine similarity between tweet and [surprise, doubt, nodoubt]
    :param tweet: raw tweet
    :return: vector of 3 cosine vectors, surpriseScore, doubtScore, nodoubtScore
    """

    tweetVector=cumulative_vector_wordList(preprocess_and_tokenize_tweet(tweet))

    #Calculate the cosine similarities
    surpriseScore=cosine(tweetVector, surpriseVector)
    doubtScore=cosine(tweetVector, doubtVector)
    noDoubtScore=cosine(tweetVector, noDoubtVector)

    return [surpriseScore,doubtScore,noDoubtScore]

