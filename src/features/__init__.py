from settings import DATA_EXTERNAL_ROOT
import os
from nltk.parse.stanford import StanfordDependencyParser

#Load Stanford NLP for negation
path_to_jar = os.path.join(DATA_EXTERNAL_ROOT,'stanford-corenlp-full-2017-06-09','stanford-corenlp-3.8.0.jar')
path_to_models_jar = os.path.join(DATA_EXTERNAL_ROOT,'stanford-corenlp-full-2017-06-09','stanford-corenlp-3.8.0-models.jar')
stanford_dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
