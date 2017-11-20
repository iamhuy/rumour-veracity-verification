import os
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_RAW_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_INTERIM_ROOT = os.path.join(PROJECT_ROOT, 'data', 'interim')
DATA_PROCESSED_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed')
DATA_EXTERNAL_ROOT = os.path.join(PROJECT_ROOT, 'data', 'external')

MODELS_ROOT = os.path.join(PROJECT_ROOT, 'models')
TRAINING_OPTIONS = ['instance-based', 'svm', 'j48', 'bayes', 'random-forest']
FEATURE_OPTIONS = {
    'social_features': ["num_followers", "favorites_score"],
    'user_features': []
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