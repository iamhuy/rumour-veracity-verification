from utils import preprocess_tweet
from src.features import google_bad_words_list, noswearing_bad_words_list, netlingo_acronyms_list


def check_existence_of_words(tweet, wordlist):
    """
    Function for the slang or curse words and acronyms features
    :param tweet: semi process tweet (hashtags mentions removed)
    :param wordlist:List of words
    :return: the binary vector of word in the tweet
    """

    tweet=preprocess_tweet(tweet)
    found_word = 0
    for word in wordlist:
        if tweet.find(word) != -1:
            found_word = 1
            break

    return [found_word]


def contain_google_bad_words(tweet):
    """
    Return whether the tweet contains google bad words or not
    :param tweet: Raw tweet
    :return: a binary vector
    """
    return check_existence_of_words(tweet, google_bad_words_list)


def contain_noswearing_bad_words(tweet):
    """
    Return whether the tweet contains noswearing.com bad words or not
    :param tweet: Raw tweet
    :return: a binary vector    """

    return check_existence_of_words(tweet, noswearing_bad_words_list)


def contain_acronyms(tweet):
    """
    Return whether the tweet contains acronyms
    :param tweet:Raw tweet
    :return:a binary vector
    """
    return check_existence_of_words(tweet, netlingo_acronyms_list)
