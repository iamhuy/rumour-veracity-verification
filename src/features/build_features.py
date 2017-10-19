from has_description import has_description
from geo_enabled import geo_enabled


def collect_feature(tweet):
    """
        Collect a set of featrues from a tweet
    :param tweet: a json object representing a tweet
    :return: A vector represents all the feature
    """

    feature_vector = []

    # Whether the user has description or not.
    feature_vector += has_description(tweet['user'])

    # Whether the user has enabled geo-location or not.
    feature_vector += geo_enabled(tweet['user'])

    return feature_vector
