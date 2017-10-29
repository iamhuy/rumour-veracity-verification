import  re

def dotdotdot(text):
    """
    Extract 2 features about dot dot dot from tweet text - hasDotDotDot(0-1) and numberOfDotDotDot
    :param text: tweet content text
    :return: vector of size 2 [ <has dot dot dot>, <number of dot dot dot>]
    """
    num_occurences = re.findall(r'\.\.\.', text)
    return [1 if len(num_occurences) > 0 else 0, len(num_occurences)]


def num_occurrences(text, pattern):
    """
    Extract number of occurences of a pattern in text
    :param text: a string represet text
    :param pattern: a regex pattern
    :return: Integer indicates number of occurrences
    """
    return len(re.findall(pattern, text))


