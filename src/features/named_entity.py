from src.features import stanford_ner
from nltk import word_tokenize

def get_named_entity(tweet):
    """
        Get the occurrences of four tags (Person, Organization, Location, Money)
    :param tweet: raw tweet
    :return: The NER vector of 4 mentioned tags
    """
    tweet_tokens = word_tokenize(tweet)
    ner_tag = stanford_ner.tag(tweet_tokens)
    person_occur = 0
    org_occur = 0
    loc_occur = 0

    for (_, rel) in ner_tag:
        if rel=="PERSON":
            person_occur+=1
        elif rel=="ORGANIZATION":
            org_occur+=1
        elif rel=="LOCATION":
            loc_occur+=1

    return [person_occur, org_occur, loc_occur]
