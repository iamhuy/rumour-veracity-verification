


def geo_enabled(user):
    """
        Check if a user enabled geo location or not
    :param tweet: a json object representing a user
    :return: A vector of size 1 : [x]
            x = 0 if the user has enabled geo location
            x = 1 otherwise
    """

    return [1] if user['geo_enabled'] != None else [0]