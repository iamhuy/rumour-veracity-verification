import json

def json_from_file(file_path):
    """
    Read a file and convert it into a json object
    :param file_path: path of file
    :return: A json object
    """
    file = open(file_path, "r")
    return json.loads(file.read())


def merge_json(a, b):
    """
    Merge 2 json objects into 1
    :param a: Json object 1
    :param b: Json object 2
    :return: Merged Json object
    """
    return dict(list(a.items()) + list(b.items()))