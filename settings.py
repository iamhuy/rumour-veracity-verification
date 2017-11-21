import os
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_RAW_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_INTERIM_ROOT = os.path.join(PROJECT_ROOT, 'data', 'interim')
DATA_PROCESSED_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed')
DATA_EXTERNAL_ROOT = os.path.join(PROJECT_ROOT, 'data', 'external')

MODELS_ROOT = os.path.join(PROJECT_ROOT, 'models')
TRAINING_OPTIONS = ['instance-based', 'svm', 'j48', 'bayes', 'random-forest']

FEATURE_LIST = [
    None,
    "brown_cluster",            #1
    "pos_tag"                   #2
    "sentiment",                #3
    "named_entity",             #4
    "is_reply",                 #5
    "emoticon",                 #6
    "has_url",                  #7
    "originality_score",        #8
    "user_verified",            #9
    "num_followers",            #10
    "role_score",               #11
    "engagement_score",         #12
    "favorites_score",          #13
    "geo_enabled",              #14
    "has_description",          #15
    "description_length",       #16
    "average_negation",         #17
    "has_negation",             #18
    "has_swearing_word",        #19
    "has_bad_word",             #20
    "has_acronyms",             #21
    "suprise_score",            #22
    "doubt_score",              #23
    "no_doubt_score",           #24
    "has_question_mark",        #25
    "has_exclamation_mark",     #26
    "has_dotdotdot",            #27
    "has_question_mark",        #28
    "has_exclamation_mark",     #29
    "has_dotdotdot",            #30
    "regex",                    #31
    "average_word_length"       #32
]

FEATURE_OPTIONS = {
    'all_features': range(1,33),
    'lexical_features': [1, 2, 4, 17, 18, 19, 20, 21],
    'sentiment_features': [3, 6,  22, 23, 24],
    'punctuation_features': [25, 26, 27, 28, 29, 30],
    'rule_based_features': [31],
    'user_features': [8, 9, 10, 11, 12, 13, 14, 15, 16],
    'tweet_features': [5, 7, 32]
}

TRAINING_SETTINGS = {
    'features_subset':'social_features',
    'balancing_class_algorithm': None,
    'scale_option': None,
    'reduce_dimension_algorithm': None,
    'training_algorithm': {
        'name': 'instance-based',
        'k': 1,
    }
}