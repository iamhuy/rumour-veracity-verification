#!/usr/bin/python
from sklearn.feature_extraction.text import CountVectorizer
import os
import json
import preprocessor as p
import sys
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import nltk
import string
import itertools

fix_dict = "/home/potus/rumour-veracity-verification/data/raw/semeval2017-task8-dataset/rumoureval-data/"


def get_text(a_file):
    input_file = open(a_file, 'r')
    tweet_dict = json.load(input_file)
    return tweet_dict['text']


def get_immediate_subdirectories(a_dir):
    print a_dir
    print(os.getcwd())
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def remove_stopwords(tweet):
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
    return ' '.join(no_stop_words)


def preprocess_tweet(tweet):
    cleaned_tweet = tweet.lower()  # lowercase the tweet
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)  # set options for the preprocessor
    cleaned_tweet = p.clean(cleaned_tweet)
    cleaned_tweet = remove_stopwords(cleaned_tweet)  # remove stopwords
    return cleaned_tweet;


def tweet_clean(tweet):
    cache_english_stopwords = set(stopwords.words('english'))
    # Remove tickers
    sent_no_tickers = re.sub(r'\$\w*', '', tweet.lower())
    # print('No tickers:')
    # print(sent_no_tickers)
    tw_tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    temp_tw_list = tw_tknzr.tokenize(sent_no_tickers)
    # print('Temp_list:')
    # print(temp_tw_list)
    # Remove stopwords
    list_no_stopwords = [i for i in temp_tw_list if i.lower() not in cache_english_stopwords]
    # print('No Stopwords:')
    # print(list_no_stopwords)
    # Remove hyperlinks
    list_no_hyperlinks = [re.sub(r'https?:\/\/.*\/\w*', '', i) for i in list_no_stopwords]
    # print('No hyperlinks:')
    # print(list_no_hyperlinks)
    # Remove hashtags
    list_no_hashtags = [re.sub(r'#\w*', '', i) for i in list_no_hyperlinks]
    # print('No hashtags:')
    # print(list_no_hashtags)
    # Remove Punctuation and split 's, 't, 've with a space for filter
    list_no_punctuation = [re.sub(r'[' + string.punctuation + ']+', ' ', i) for i in list_no_hashtags]
    # print('No punctuation:')
    # print(list_no_punctuation)
    # Remove multiple whitespace
    new_sent = ' '.join(list_no_punctuation)
    # Remove any words with 2 or fewer letters
    filtered_list = tw_tknzr.tokenize(new_sent)
    list_filtered = [re.sub(r'^\w\w?$', '', i) for i in filtered_list]
    # print('Clean list of words:')
    # print(list_filtered)
    filtered_sent = ' '.join(list_filtered)
    clean_sent = re.sub(r'\s\s+', ' ', filtered_sent)
    # Remove any whitespace at the front of the sentence
    clean_sent = clean_sent.lstrip(' ')
    # print('Clean sentence:')
    # print(clean_sent)
    return clean_sent


def create_corpus_for_story(current_story):
    corpus = []
    sub_direc = get_immediate_subdirectories(fix_dict+current_story+"/")
    for direct in sub_direc:
        # tweet = preprocess_tweet( get_text(current_story+"/"+direct+"/source-tweet/"+direct+".json").encode('ascii','ignore') ).encode('ascii','ignore')

        tweet = tweet_clean(
            get_text(fix_dict + current_story + "/" + direct + "/source-tweet/" + direct + ".json").encode('ascii',
                                                                                                           'ignore')).encode(
            'ascii', 'ignore')
        corpus.append(tweet)
    return corpus

def main():
    #current_story = sys.agrv[1]
    #prepare the vector
    tag_set=['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    bigram_tag=[]
    for i in itertools.product(tag_set, tag_set):
        bigram_tag.append(str(i))
    #print (bigram_tag.index("('ADJ', 'ADJ')"))
    #create corpus
    corpus = create_corpus_for_story("ebola-essien")
    print ( corpus[:10] )
    tagged_tokens=[]
    for tweet in corpus:
        token=nltk.word_tokenize(tweet)
        tagged_token=nltk.pos_tag(token,tagset="universal")
        tagged_tokens.append(tagged_token)
        print (tagged_token)
        #create the vector size of bigram_tag
        pos_vector=[0]*len(bigram_tag)
        for i in range(0, (len(tagged_token)-1)):
            #print ( tagged_token[i][1]+","+tagged_token[i+1][1] )
            pos_vector[( bigram_tag.index(str("('"+tagged_token[i][1]+"'"+", "+"'"+tagged_token[i+1][1]+"')")) )] = 1

        print (pos_vector)
    #print ( tagged_tokens )
if __name__== "__main__":
    main()
