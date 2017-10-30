import json
import math
import os.path
import sys
from src.data.constants import VERACITY_LABELS


def sem_eval_score(actuals, golds):

    correct = 0
    total = len(golds)
    errors = []

    for idx, (actual_value, confidence) in enumerate(actuals):
        gold_value = golds[idx]

        if actual_value == gold_value:
            correct += 1
            errors.append((1 - confidence) ** 2)

        elif VERACITY_LABELS[actual_value] == 'unverified':
            errors.append((confidence) ** 2)

        else:
            errors.append(1.0)


    score = float(correct) / float(total)
    rmse = math.sqrt(sum(errors) / len(errors))

    print('veracity accuracy:', score)
    print('confidence rmse:  ', rmse)