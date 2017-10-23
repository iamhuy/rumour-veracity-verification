from num_occurrences import num_occurrences
from user_features import user_verified, has_description, geo_enabled, num_followers


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

    # Whether the user is verified or not
    feature_vector += user_verified(tweet['user'])

    # Number of followers
    feature_vector += num_followers(tweet['user'])

    # Whether the tweet contain dot dot dot or not and number of dot dot dot
    dotdotdot_occurrences = num_occurrences(tweet['text'], r'\.\.\.')
    feature_vector += [1 if dotdotdot_occurrences > 0 else 0, dotdotdot_occurrences]

    # Whether the tweet contain exclamation mark or not and number of exclamation marks
    exclamation_mark_occurrences = num_occurrences(tweet['text'], r'!')
    feature_vector += [1 if exclamation_mark_occurrences > 0 else 0, exclamation_mark_occurrences]

    # Whether the tweet contain question mark or not and number of question marks
    question_mark_occurrences = num_occurrences(tweet['text'], r'?')
    feature_vector += [1 if question_mark_occurrences > 0 else 0, question_mark_occurrences]


    return feature_vector
