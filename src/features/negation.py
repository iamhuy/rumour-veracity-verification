from nltk.parse.stanford import StanfordDependencyParser
def get_average_negation(tweet):
    path_to_jar = '/home/potus/rumour-veracity-verification/data/external/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar'
    path_to_models_jar = '/home/potus/rumour-veracity-verification/data/external/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0-models.jar'
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    result = dependency_parser.raw_parse(tweet)
    dep = result.next()
    dependecies=list(dep.triples())
    count_negation=0;
    for (_,rel,_) in dependecies:
        if rel=='neg':
            count_negation=count_negation+1
            #print "To here"
    #print count_negation
    #print len(dependecies)
    average=float(count_negation)/len(dependecies)
    return average