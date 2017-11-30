import os
from sklearn.feature_selection import *
from sklearn.ensemble import *

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_RAW_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_INTERIM_ROOT = os.path.join(PROJECT_ROOT, 'data', 'interim')
DATA_PROCESSED_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed')
DATA_EXTERNAL_ROOT = os.path.join(PROJECT_ROOT, 'data', 'external')
MODELS_ROOT = os.path.join(PROJECT_ROOT, 'models')

FEATURE_LIST = [
    None,
    "brown_cluster",            #1
    None,                       #2
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
    "average_word_length",      #32
    "pos_tag_1gram",            #33
    "pos_tag_2gram",            #34
    "pos_tag_3gram",            #35
    "pos_tag_4gram",            #36
    "stance"                    #37
]

one_to_37 = range(1,38)
one_to_37.remove(2)

FEATURE_OPTIONS = {
    'all_features': one_to_37,
    'lexical_features': [1, 33, 34, 35, 36, 4, 4, 17, 18, 19, 20, 21],
    'sentiment_features': [3, 6,  22, 23, 24, 37],
    'punctuation_features': [25, 26, 27, 28, 29, 30],
    'rule_based_features': [31],
    'user_features': [8, 9, 10, 11, 12, 13, 14, 15, 16],
    'user+tweet_features': [8, 9, 10, 11, 12, 13, 14, 15, 16, 5, 7, 32],
    'tweet_features': [5, 7, 32]
}

TRAINING_SETTINGS = {
    'features_subset': 'all_features',

    'balancing_class_algorithm': {
        'name': 'SMOTE',
        'k': 4
    },
    # 'balancing_class_algorithm': None,


    'scale_option': {
        'name': 'MinMax',
    },
    # 'scale_option': None,


    'reduce_dimension_algorithm': {
        'name' : 'PCA',
        'n_components': 50
    },
    # 'reduce_dimension_algorithm': None,


    'feature_selection_algorithm' : {
        'name': 'k-best',
        'score_func': chi2,
        'k': 100
        # 'threshold': 0.25,
        # 'prefit': True,
        # 'model':ExtraTreesClassifier()
        # 'alpha': 0.1
    },
    # 'feature_selection_algorithm': None,


    'training_algorithm': {
        'name': 'gradient-boosting',
        'random_state': 0,
        # 'class_weight': {0:1,1:10,1:1},
        'n_estimators': 100,
        # 'bootstrap': True,
        # 'verbose': True
        'learning_rate': 0.1
    }
}