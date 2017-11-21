from src.features import stanford_dependency_parser
def get_average_negation(tweet):
    """
    Return the average negation and the bin value of having negation relation based on Stanford Parser
    :param tweet: raw tweet
    :return: the average of negation on total of relations and the binary of having negation or not
    """
    result = stanford_dependency_parser.raw_parse(tweet)
    dep = result.next()
    dependencies = list(dep.triples())
    average=0
    hasNegation=0
    if len(dependencies) > 0:
        count_negation = 0;
        for (_, rel, _) in dependencies:
            if rel == 'neg':
                count_negation += 1
        if count_negation != 0:
            hasNegation=1
        average = float(count_negation)/len(dependencies)

    return [average, hasNegation]