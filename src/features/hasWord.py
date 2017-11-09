from utils import preprocess_tweet

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
