from dateutil import parser
import preprocessor as p

def timestamp_to_date(timestamp):
    """
        Conver a twitter timestamp to a datetime object
    :param timestamp: a string represent the timestamp
    :return: a datetime object
    """

    return parser.parse(timestamp)


def day_diff(timestamp1, timestamp2):
    """
        Number of days between 2 timestamps
    :param timestamp1: first timestamp
    :param timestamp2: second timestamp
    :return: An integer indicating number of days between 2 timestamps
    """

    return (timestamp_to_date(timestamp1) - timestamp_to_date(timestamp2)).days

def preprocess_tweet(tweet):
    """
    Clean the tweet before feeding to other functions
    :param tweet: a raw tweet
    :return: tweet with URL, MENTIONS, EMOJI, HASTHTAGS removed
    """
    cleaned_tweet = tweet.lower()  # lowercase the tweet
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)  # set options for the preprocessor
    cleaned_tweet = p.clean(cleaned_tweet.encode("ascii", "ignore"))
    return cleaned_tweet;