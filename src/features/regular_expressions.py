import re
from utils import preprocess_tweet

def regex_vector(tweet):
    """
    Return the binary regex vector of the tweet
    :param tweet: raw tweet
    :return: the vector in which each bit represent the existence of this regex
    """
    tweet=preprocess_tweet(tweet)
    patterns = ["is (this|that|it) true", "wh[a]*t[?!][?1]*", "(real?|really?|unconfirmed)", "(rumour|debunk)",
                "(that|this|it) is not true"]
    patterns_vector = [0] * len(patterns)
    pattern_compiled = map(re.compile,patterns)
    for i in range(0, len(pattern_compiled)):
        if pattern_compiled[i].findall(tweet):
            patterns_vector[i] = 1

    return patterns_vector