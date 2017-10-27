#!/usr/bin/python
from sklearn.feature_extraction.text import CountVectorizer
import os
import json
import preprocessor as p
import sys
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import string
import numpy as np

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
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG)  # set options for the preprocessor
    cleaned_tweet = p.clean(cleaned_tweet)
    #cleaned_tweet = remove_stopwords(cleaned_tweet)  # remove stopwords
    return cleaned_tweet;

def get_emoticons_vectors(tweet):
    """
    Return 23 types of emoticons based on Wikipedia
    :param tweet:a non preprocessing tweet
    :return:a 23-long binary vector indicating which types of emoticon existing in the tweet
    """
    smiley={":-)",":)",":]",":-]",":3",":-3",":>",":->","8)","8-)",":}",":-}",":o)",":c)",":^)","=]","=)"}
    laughing={":-D",":D","8-D","8D","x-D","xD","X-D","XD","=D","=3","B^D"}
    very_happy={":-))"}
    sad={":-(",":(",":c",":-c",":<",":-<",":[",":-[",":-||",">:[",":{",":@",">:("}
    crying={":-'(",":'("}
    tears_of_happy={":'-)",":')"}
    horror={"D-':","D:<","D:","D8","D;","D=","DX"}
    suprise={":-O",":O",":o",":-o",":-0","8-0",">:O"}
    kiss={":-*",":*",":x"}
    wink={";-)",";)","*-)","*)",";-]",";]",";^)",":-,",";D"}
    toungue={":-P",":P","X-P","XP","x-p","xp",":-p",":p",":-b",":b","d:","=p",">:P"} #incomple
    skeptical={":-/",":/",":-.",">:\\",">:/",":\\","=/","=\\",":L","=L",":S"}
    indecision={":|",":-|"}
    embarrassed={":$"}
    sealed_lips={":-X",":X",":-#",":#",":-&",":&"}
    innocent={"O:)","O:-)","0:3","0:-3","0:-)","0:)","0;^)"}
    evil={">:)",">:-)","}:)","}:-)","3:)","3:-)",">;)"}
    cool={"|;-)","|-O"}
    tongue_in_cheek={":-J"}
    parited={"#-)"}
    confused={"%-)","%)"}
    sick={":-###..",":###.."}
    dumb={"<:-|"}

    words=set(tweet.split())
    if sick & words:
        print "So sick"
    emoticon_vectors=[bit_wise_to_bool(smiley, words), bit_wise_to_bool(laughing, words), bit_wise_to_bool(very_happy, words),bit_wise_to_bool(sad, words),
    bit_wise_to_bool(crying, words),bit_wise_to_bool(tears_of_happy, words),bit_wise_to_bool(horror, words),bit_wise_to_bool(suprise, words),bit_wise_to_bool(kiss, words),bit_wise_to_bool(wink, words),
    bit_wise_to_bool(toungue, words),bit_wise_to_bool(skeptical, words),bit_wise_to_bool(indecision, words),bit_wise_to_bool(embarrassed, words),bit_wise_to_bool(sealed_lips, words),bit_wise_to_bool(innocent, words),
    bit_wise_to_bool(evil, words),bit_wise_to_bool(cool, words),bit_wise_to_bool(tongue_in_cheek, words),bit_wise_to_bool(parited, words),bit_wise_to_bool(confused, words),bit_wise_to_bool(sick, words),
    bit_wise_to_bool(dumb, words)]
    return emoticon_vectors
def bit_wise_to_bool(a, b):
    tmp=0
    if a & b:
        tmp=1
    else:
        tmp=0
    return tmp
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
    list_no_punctuation = [re.sub(r'[ ' + string.punctuation + ' ]+', ' ', i) for i in list_no_hashtags]
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
        #tweet = preprocess_tweet( get_text(current_story+"/"+direct+"/source-tweet/"+direct+".json").encode('ascii','ignore') ).encode('ascii','ignore')

        tweet = (
            get_text(fix_dict + current_story + "/" + direct + "/source-tweet/" + direct + ".json").encode('ascii',
                                                                                                           'ignore')).encode(
           'ascii', 'ignore')
        corpus.append(tweet)
    return corpus

def main():
    #current_story = sys.agrv[1]
    corpus = create_corpus_for_story("ottawashooting")
    for tweet in  corpus:
        print tweet
        print get_emoticons_vectors(tweet)

if __name__== "__main__":
    main()
