from utils import day_diff


def user_verified(user):
    """
    Check if a twitter user has been verified or not
    :param user: a json object rerpresent user
    :return: binary vector of size-1 [<user is verified>]
    """
    return [1 if user['verified'] else 0]


def geo_enabled(user):
    """
        Check if a user enabled geo location or not
    :param tweet: a json object representing a user
    :return: A vector of size 1 : [x]
            x = 0 if the user has enabled geo location
            x = 1 otherwise
    """

    return [1] if user['geo_enabled'] != None else [0]


def has_description(user):
    """
        Check if a user of a twitter has description or not
    :param tweet: a json object representing a user
    :return: A vector of size 1 : [x]
            x = 0 if the user has profile description
            x = 1 otherwise
    """

    return [1] if user['description'] != None else [0]


def num_followers(user):
    """
        Number of followers that a twitter user follows
    :param user:  a json object representing a user
    :return: a vector of size 1 [<number of followers>]
    """

    return [user['followers_count']]


def originality_score(user):
    """
        Number of statues that a twitter user has posted
    :param user: a json object representing a user
    :return: a vector if size 1 [<number of statuses count>]
    """

    return [user['statuses_count']]


def role_score(user):
    """
        Ratio between the number of followers and followees
    :param user: a json object representing a user
    :return: a vector if size 1 [<Ratio between the number of followers and followees>]
            if number of followees is 0, score is assigned with 1,000,000
    """

    return [float(user['followers_count']) / float(user['friends_count']) if user['friends_count'] > 0 else 1000000]


def engagement_score(tweet):
    """
        Ratio between the number of tweets and number of active days
    :param user: a json object representing a tweet
    :return: a vector if size 1 [<Ratio between the number of tweets and number of active days>]
            if number of active days is 0, score is assigned with 1,000,000
    """


    active_days = day_diff(tweet['created_at'], tweet['user']['created_at'])
    return [float(tweet['user']['statuses_count']) / float(active_days) if active_days > 0 else 1000000]


def favorites_score(tweet):
    """
        Ratio between the number of favorites and number of active days
    :param user: a json object representing a tweet
    :return: a vector if size 1 [<Ratio between the number of favorites and number of active days>]
            if number of active days is 0, score is assigned with 1,000,000
    """

    active_days = day_diff(tweet['created_at'], tweet['user']['created_at'])
    return [float(tweet['user']['favourites_count']) / float(active_days) if active_days > 0 else 1000000]