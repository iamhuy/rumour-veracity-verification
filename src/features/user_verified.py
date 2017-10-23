

def user_verified(user):
    """
    Check if a tweet user has been verified or not
    :param user: a json object rerpresent user
    :return: binary vector of size-1 [<user is verified>]
    """
    return [1 if user['verified'] else 0]