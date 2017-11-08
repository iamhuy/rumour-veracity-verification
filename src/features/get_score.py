import requests
import json
import re
import gensim
import logging
import numpy as np
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from scipy.spatial.distance import cosine


def remove_stopwords_and_tokenize(tweet):
    stop_words = set(stopwords.words('english'))
    tweet = tweet.lower()
    # Remove stopwords
    tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    tweet_tokens = tweet_tokenizer.tokenize(tweet)
    no_stop_words = []
    for token in tweet_tokens:
        if not token in stop_words:
            no_stop_words.append(token)
            # remove words less than 2 letters
            
    no_stop_words = [re.sub(r'^\w\w?$', '', i) for i in no_stop_words]
    return no_stop_words


def preprocess_and_tokenize_tweet(tweet):
    cleaned_tweet = tweet.lower()  # lowercase the tweet
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)  # set options for the preprocessor
    cleaned_tweet = p.clean(cleaned_tweet.encode('ascii', 'ignore'))
    cleaned_tweet = remove_stopwords_and_tokenize(cleaned_tweet)  # remove stopwords
    return cleaned_tweet;


def get_wordList(feedingWord):
    app_id = '93a6b0a3'
    app_key = 'a8bab0b9458264690625fdd834421d10'

    language = 'en'

    url = 'https://od-api.oxforddictionaries.com:443/api/v1/entries/' + language + '/' + feedingWord.lower() + '/synonyms;antonyms'

    r = requests.get(url, headers={'app_id': app_id, 'app_key': app_key})

    # print("text \n" + r.text)
    pattern = re.compile("(^(?!.*[-])(\w)+$)")
    # print("json \n" + json.dumps(r.json()))
    j = json.loads(r.text)
    wordList = []
    for syn in (j["results"][0]["lexicalEntries"][0]["entries"][0]["senses"][0]["synonyms"]):
        if (pattern.match(syn["text"])):
            wordList.append(syn["text"])
    return wordList


def cummulative_vector_wordList(wordList):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        numOfWord = len(wordList)
        # Load Google's pre-trained Word2Vec model.
        model = gensim.models.KeyedVectors.load_word2vec_format(
            "/media/potus/New volume/potus/Downloads/GoogleNews-vectors-negative300.bin", binary=True, limit=1000000)
        cummulative_vector = np.zeros((300,), dtype=np.float)
        for word in wordList:
            try:
                cummulative_vector = np.add(cummulative_vector, model[word])
            except KeyError:
                numOfWord -= 1
                continue
        if numOfWord == 0:
            return None
        else:
            #print numOfWord
            #print cummulative_vector
            return np.divide(cummulative_vector, numOfWord)


def get_Vector(tweet):
    surpriseVector=cummulative_vector_wordList(get_wordList("surprised"))
    doubtVector=cummulative_vector_wordList(get_wordList("uncertain"))
    noDoubtVector=cummulative_vector_wordList(get_wordList("sure"))
    tweetVector=cummulative_vector_wordList(preprocess_and_tokenize_tweet(tweet))
    surpriseScore=cosine(tweetVector, surpriseVector)
    doubtScore=cosine(tweetVector, doubtVector)
    noDoubtScore=cosine(tweetVector, noDoubtVector)
    return [surpriseScore,doubtScore,noDoubtScore]
def main():
	print(get_Vector("What a surprise! There is no doubt"))
if __name__== "__main__":
    main()

