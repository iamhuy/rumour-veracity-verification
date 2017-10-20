import os
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_RAW_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_INTERIM_ROOT = os.path.join(PROJECT_ROOT, 'data', 'interim')
DATA_PROCESSED_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_ROOT = os.path.join(PROJECT_ROOT, 'models')

TRAINING_OPTIONS = ['instance-based', 'svm', 'j48', 'bayes', 'random-forest']