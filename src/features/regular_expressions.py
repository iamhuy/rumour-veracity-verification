import re
import preprocessor as p

def preprocess_tweet(tweet):
    cleaned_tweet = tweet.lower()  # lowercase the tweet
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)  # set options for the preprocessor
    cleaned_tweet = p.clean(cleaned_tweet.encode("ascii", "ignore"))
    return cleaned_tweet;
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
    for i in range(0, len(patterns)):
        pattern = re.compile(patterns[i])
        if pattern.findall(tweet):
            patterns_vector[i] = 1
    #print(patterns_vector)
    return patterns_vector