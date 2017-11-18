from settings import MODELS_ROOT
import os
import json


feature_bitmask_file = open(os.path.join(MODELS_ROOT, 'feature_bitmask'), "r")
feature_bitmask = json.loads(feature_bitmask_file.read())