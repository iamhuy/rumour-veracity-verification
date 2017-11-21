#!/usr/bin/python
import preprocessor as p
from nltk.tokenize import TweetTokenizer


def preprocess_tweet(tweet):
    #p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG)  # set options for the preprocessor
    p.set_options(p.OPT.URL, p.OPT.MENTION)  # set options for the preprocessor
    cleaned_tweet = p.clean(tweet)
    return cleaned_tweet

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

    #words=set(tweet.split())
    tweet=preprocess_tweet(tweet)
    tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    words = set(tweet_tokenizer.tokenize(tweet))

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
