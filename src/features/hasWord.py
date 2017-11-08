import pickle
import preprocessor as p


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


def preprocess_tweet(tweet):

    cleaned_tweet = tweet.lower()  # lowercase the tweet
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)  # set options for the preprocessor
    cleaned_tweet = p.clean(cleaned_tweet.encode("ascii", "ignore"))

    return cleaned_tweet;


def check_existence_of_words(tweet, wordlist):
    """
    Function for the slang or curse words and acronyms features
    :param tweet: semi process tweet (hashtags mentions removed)
    :param wordlist:List of words
    :return: the binary vector of word in the tweet
    """

    tweet=preprocess_tweet(tweet)
    boolean=0
    for word in wordlist:
        if (tweet.find(word) != -1):
            boolean=1
            break

    return [boolean]
