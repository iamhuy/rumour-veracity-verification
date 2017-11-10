from num_occurrences import num_occurrences
from user_features import *
from pos_tag import get_bigram_postag_vector, get_trigram_postag_vector
from sentiment_StanfordNLP import get_sentiment_value
from src.data.constants import STANCE_LABELS_MAPPING
from emoticon import get_emoticons_vectors
from brown_cluster import brown_cluster
from has_word import contain_noswearing_bad_words, contain_acronyms, contain_google_bad_words


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

    # Number of statuses of user
    feature_vector += originality_score(tweet['user'])

    # Role score
    feature_vector += role_score(tweet['user'])

    # Engagement score
    feature_vector += engagement_score(tweet)

    # Favourites score
    feature_vector += favorites_score(tweet)

    # Is a reply or not
    feature_vector += [1 if tweet['in_reply_to_status_id'] != None else 0]

    # Whether the tweet contain dot dot dot or not and number of dot dot dot
    dotdotdot_occurrences = num_occurrences(tweet['text'], r'\.\.\.')
    feature_vector += [1 if dotdotdot_occurrences > 0 else 0, dotdotdot_occurrences]

    # Whether the tweet contain exclamation mark or not and number of exclamation marks
    exclamation_mark_occurrences = num_occurrences(tweet['text'], r'!')
    feature_vector += [1 if exclamation_mark_occurrences > 0 else 0, exclamation_mark_occurrences]

    # Whether the tweet contain question mark or not and number of question marks
    question_mark_occurrences = num_occurrences(tweet['text'], r'\?')
    feature_vector += [1 if question_mark_occurrences > 0 else 0, question_mark_occurrences]

    # Brown clusters
    brown_cluster_vector, has_url = brown_cluster(tweet['text'])
    feature_vector += brown_cluster_vector

    # Contain URL feature
    feature_vector += [1 if has_url else 0]

    # Postag features
    feature_vector += get_bigram_postag_vector(tweet['text'])
    feature_vector += get_trigram_postag_vector(tweet['text'])

    # Sentiment features
    feature_vector += get_sentiment_value(tweet['text'])

    # Stance features
    feature_vector += [STANCE_LABELS_MAPPING[tweet['stance']]]

    # Emoticon feature
    feature_vector += get_emoticons_vectors(tweet['text'])

    # Has Acronyms
    feature_vector += contain_acronyms(tweet['text'])

    # Has bad words
    feature_vector += contain_google_bad_words(tweet['text'])

    # Has no swearing bad words
    feature_vector += contain_noswearing_bad_words(tweet['text'])


    return feature_vector
