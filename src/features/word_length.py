def description_length(user):
    """
        Get the length of user description in words
    :param user: the user object in json tweet
    :return: vector length of 1 indicating the length of description. If user
            if user has no description, [0] will be returned
    """
    des_length=0
    if user['description']:
        description = user['description']
        des_length = len(description.split())
    return des_length


def average_word_length(tweet):
    """
        Return the average length of the tweet
    :param tweet: raw text tweet
    :return: the float number of character count divided by the word count
    """
    character_count = 0
    word_count = 0
    for c in tweet:
        character_count += 1
    word_count = len(tweet.split())
    return float(character_count)/float(word_count)