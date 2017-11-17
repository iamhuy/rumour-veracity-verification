from num_occurrences import num_occurrences
from user_features import *
from pos_tag import get_ngram_postag_vector
from sentiment_StanfordNLP import get_sentiment_value
from src.data.constants import STANCE_LABELS_MAPPING
from emoticon import get_emoticons_vectors
from brown_cluster import brown_cluster
from has_word import contain_noswearing_bad_words, contain_acronyms, contain_google_bad_words
from regular_expressions import regex_vector
from get_score import get_vectors
from word_length import average_word_length, description_length
from named_entity import get_named_entity
from settings import MODELS_ROOT
import json
import os

feature_bitmask = dict()

def collect_feature(tweet):
    """
        Collect a set of featrues from a tweet
    :param tweet: a json object representing a tweet
    :return: A vector represents all the feature
    """

    feature_vector = []

    # Whether the user has description or not.
    pivot = len(feature_vector)
    feature_vector += has_description(tweet['user'])
    feature_bitmask['has_description'] = (pivot, len(feature_vector))

    # Length of user description in words
    pivot = len(feature_vector)
    feature_vector += [description_length(tweet['user'])]
    feature_bitmask['description_length'] = (pivot, len(feature_vector))

    # Average length of a word
    pivot = len(feature_vector)
    feature_vector += [average_word_length(tweet['text'])]
    feature_bitmask['average_word_length'] = (pivot, len(feature_vector))

    # Whether the user has enabled geo-location or not.
    pivot = len(feature_vector)
    feature_vector += geo_enabled(tweet['user'])
    feature_bitmask['geo_enabled'] = (pivot, len(feature_vector))

    # Whether the user is verified or not
    pivot = len(feature_vector)
    feature_vector += user_verified(tweet['user'])
    feature_bitmask['user_verified'] = (pivot, len(feature_vector))

    # Number of followers
    pivot = len(feature_vector)
    feature_vector += num_followers(tweet['user'])
    feature_bitmask['num_followers'] = (pivot, len(feature_vector))

    # Number of statuses of user
    pivot = len(feature_vector)
    feature_vector += originality_score(tweet['user'])
    feature_bitmask['originality_score'] = (pivot, len(feature_vector))

    # Role score
    pivot = len(feature_vector)
    feature_vector += role_score(tweet['user'])
    feature_bitmask['role_score'] = (pivot, len(feature_vector))

    # Engagement score
    pivot = len(feature_vector)
    feature_vector += engagement_score(tweet)
    feature_bitmask['engagement_score'] = (pivot, len(feature_vector))

    # Favourites score
    pivot = len(feature_vector)
    feature_vector += favorites_score(tweet)
    feature_bitmask['favorites_score'] = (pivot, len(feature_vector))

    # Is a reply or not
    pivot = len(feature_vector)
    feature_vector += [1 if tweet['in_reply_to_status_id'] != None else 0]
    feature_bitmask['is_reply'] = (pivot, len(feature_vector))

    # Whether the tweet contain dot dot dot or not and number of dot dot dot
    pivot = len(feature_vector)
    dotdotdot_occurrences = num_occurrences(tweet['text'], r'\.\.\.')
    feature_vector += [1 if dotdotdot_occurrences > 0 else 0, dotdotdot_occurrences]
    # feature_vector += [1 if dotdotdot_occurrences > 0 else 0]
    feature_bitmask['has_dotdotdot'] = (pivot, len(feature_vector))

    # Whether the tweet contain exclamation mark or not and number of exclamation marks
    pivot = len(feature_vector)
    exclamation_mark_occurrences = num_occurrences(tweet['text'], r'!')
    feature_vector += [1 if exclamation_mark_occurrences > 0 else 0, exclamation_mark_occurrences]
    # feature_vector += [1 if exclamation_mark_occurrences > 0 else 0]
    feature_bitmask['has_exclamation_mark'] = (pivot, len(feature_vector))

    # Whether the tweet contain question mark or not and number of question marks
    pivot = len(feature_vector)
    question_mark_occurrences = num_occurrences(tweet['text'], r'\?')
    feature_vector += [1 if question_mark_occurrences > 0 else 0, question_mark_occurrences]
    # feature_vector += [1 if question_mark_occurrences > 0 else 0]
    feature_bitmask['has_question_mark'] = (pivot, len(feature_vector))

    # Brown clusters
    pivot = len(feature_vector)
    brown_cluster_vector, has_url = brown_cluster(tweet['text'])
    feature_vector += brown_cluster_vector
    feature_bitmask['brown_cluster'] = (pivot, len(feature_vector))

    # Contain URL feature
    pivot = len(feature_vector)
    feature_vector += [1 if has_url else 0]
    feature_bitmask['has_url'] = (pivot, len(feature_vector))

    # Postag features
    pivot = len(feature_vector)
    feature_vector += get_ngram_postag_vector(tweet['text'], 1)
    feature_bitmask['pos_tag_1gram'] = (pivot, len(feature_vector))
    pivot = len(feature_vector)
    feature_vector += get_ngram_postag_vector(tweet['text'], 2)
    feature_bitmask['pos_tag_2gram'] = (pivot, len(feature_vector))
    pivot = len(feature_vector)
    feature_vector += get_ngram_postag_vector(tweet['text'], 3)
    feature_bitmask['pos_tag_3gram'] = (pivot, len(feature_vector))
    pivot = len(feature_vector)
    feature_vector += get_ngram_postag_vector(tweet['text'], 4)
    feature_bitmask['pos_tag_4gram'] = (pivot, len(feature_vector))

    # Sentiment features
    # sentiment_vector
    pivot = len(feature_vector)
    feature_vector += get_sentiment_value(tweet['text'])
    feature_bitmask['sentiment'] = (pivot, len(feature_vector))

    # Stance features
    pivot = len(feature_vector)
    stance_vector = [0,0,0,0]
    stance_vector[STANCE_LABELS_MAPPING[tweet['stance']]] = 1
    feature_vector += stance_vector
    feature_bitmask['stance'] = (pivot, len(feature_vector))


    # Emoticon feature
    pivot = len(feature_vector)
    feature_vector += get_emoticons_vectors(tweet['text'])
    feature_bitmask['emoticon'] = (pivot, len(feature_vector))

    # Has Acronyms
    pivot = len(feature_vector)
    feature_vector += contain_acronyms(tweet['text'])
    feature_bitmask['has_acronym'] = (pivot, len(feature_vector))

    # Has bad words
    pivot = len(feature_vector)
    feature_vector += contain_google_bad_words(tweet['text'])
    feature_bitmask['has_bad_word'] = (pivot, len(feature_vector))

    # Has no swearing bad words
    pivot = len(feature_vector)
    feature_vector += contain_noswearing_bad_words(tweet['text'])
    feature_bitmask['has_swearing_word'] = (pivot, len(feature_vector))

    # Regex
    pivot = len(feature_vector)
    feature_vector += regex_vector(tweet['text'])
    feature_bitmask['regex'] = (pivot, len(feature_vector))

    # Doubt Score, No Doubt Score, Surprise score
    pivot = len(feature_vector)
    feature_vector += get_vectors(tweet['text'])
    feature_bitmask['suprise_score'] = (pivot, len(feature_vector))

    # Get Named Entity Recognition
    pivot = len(feature_vector)
    feature_vector += get_named_entity(tweet['text'])
    feature_bitmask['named_entity'] = (pivot, len(feature_vector))

    with open(os.path.join(MODELS_ROOT, 'feature_bitmask'), 'w') as outfile:
        json.dump(feature_bitmask, outfile, indent=4)
    raw_input()

    return feature_vector
