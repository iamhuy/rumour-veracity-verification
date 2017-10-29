from dateutil import parser


def timestamp_to_date(timestamp):
    """
        Conver a twitter timestamp to a datetime object
    :param timestamp: a string represent the timestamp
    :return: a datetime object
    """

    return parser.parse(timestamp)


def day_diff(timestamp1, timestamp2):
    """
        Number of days between 2 timestamps
    :param timestamp1: first timestamp
    :param timestamp2: second timestamp
    :return: An integer indicating number of days between 2 timestamps
    """

    return (timestamp_to_date(timestamp1) - timestamp_to_date(timestamp2)).days

