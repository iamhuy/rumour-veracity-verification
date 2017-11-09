import gensim
import os
import requests
import json
import re
from src.features import google_word2Vec_model
from settings import DATA_EXTERNAL_ROOT

# Load Google's pre-trained Word2Vec model.

google_word2Vec_path=os.path.join(DATA_EXTERNAL_ROOT, 'GoogleNews-vectors-negative300.bin')
google_word2Vec_model = gensim.models.KeyedVectors.load_word2vec_format(google_word2Vec_path,
                                                                        binary=True, limit=1000000)



# Load the wordList of surprise, doubt, nodoubt

def get_wordlist(feedingword):
    """
    Get the list of synonyms based on the feeding Word
    :param feedingword: the word needed to find the synonyms
    :return: list of synonyms
    """
    app_id = '93a6b0a3'
    app_key = 'a8bab0b9458264690625fdd834421d10'

    language = 'en'

    url = 'https://od-api.oxforddictionaries.com:443/api/v1/entries/' + language + '/' + feedingword.lower() + '/synonyms;antonyms'

    r = requests.get(url, headers={'app_id': app_id, 'app_key': app_key})

    #regex for choosing just the single word (no hyphenated word, no phrase)
    pattern = re.compile("(^(?!.*[-])(\w)+$)")

    j = json.loads(r.text)
    wordList = [feedingword]
    for syn in (j["results"][0]["lexicalEntries"][0]["entries"][0]["senses"][0]["synonyms"]):
        if pattern.match(syn["text"]):
            wordList.append(syn["text"])
    return wordList


surpriseList = get_wordlist("surprised")
doubtList = get_wordlist("uncertain")
noDoubtList = get_wordlist("sure")