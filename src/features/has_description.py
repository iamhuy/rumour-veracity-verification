


def has_description(user):
    """
        Check if a user of a tweet has description or not
    :param tweet: a json object representing a user
    :return: A vector of size 1 : [x]
            x = 0 if the user has profile description
            x = 1 otherwise
    """

    return [1] if user['description'] != None else [0]