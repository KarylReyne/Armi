import xmltodict
from nltk import NaiveBayesClassifier, classify
from argminer_features import get_postag_vector
from HanTa import HanoverTagger as ht
from processing.thread_manager import ThreadManager
from nltk.metrics import ConfusionMatrix


def read_corpora(collapse_claims=True):
    def get_lbl(lbl):
        if collapse_claims and (lbl == 'ClaimPro' or lbl == 'ClaimContra'):
            return 'Claim'
        return lbl

    with open("THF_corpus_v1.0/3.corpus/subtaskB_train.xml", 'r', encoding="utf-8") as xml:
        string = xml.read()
        tmp = xmltodict.parse(string)
        train = []
        [train.append([dict['Text'], get_lbl(dict['Label'])]) for dict in tmp["root"]["Sentence"]]
    with open("THF_corpus_v1.0/3.corpus/subtaskB_test.xml", 'r', encoding="utf-8") as xml:
        string = xml.read()
        tmp = xmltodict.parse(string)
        test = []
        [test.append([dict['Text'], get_lbl(dict['Label'])]) for dict in tmp["root"]["Sentence"]]
    return train, test


def map_feature_extraction(extraction_function):
    train_corpus, test_corpus = read_corpora()
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')

    tm = ThreadManager()

    postag_vector = {}
    tm.execute_as_new_process(id="get_postag_vector",
                              target=get_postag_vector,
                              args=(postag_vector, train_corpus, tagger,))
    train_set = []
    tm.execute_as_new_process(id="get_train_set",
                              target=(lambda tc, p, t:
                                      [train_set.append((extraction_function(d, p, t), a)) for (d, a) in tc]),
                              args=(train_corpus, postag_vector, tagger,))
    test_set = []
    tm.execute_as_new_process(id="get_test_set",
                              target=(lambda tc, p, t:
                                      [test_set.append((extraction_function(d, p, t), a)) for (d, a) in tc]),
                              args=(test_corpus, postag_vector, tagger,))

    return test_set, train_set


def evaluate_feature_extraction(_extraction_function, _classifier=None, header='\nNaive Bayes Classifier'):
    def shorten_label(string):
        short = "default"
        if string == 'MajorPosition': short = 'MP'
        elif string == "Premise": short = 'P'
        elif string == 'Claim': short = 'C'
        return short

    test_set, train_set = map_feature_extraction(extraction_function=_extraction_function)

    classifier = NaiveBayesClassifier.train(train_set) if _classifier is None else _classifier(train_set)
    accuracy = classify.accuracy(classifier, test_set)
    print(header)
    print("Accuracy: ", accuracy)
    classifier.show_most_informative_features(20) if _classifier is None else print(end='')
    gold = [shorten_label(l) for (f, l) in test_set]
    test = [shorten_label(classifier.classify(f)) for (f, l) in test_set]
    print(ConfusionMatrix(gold, test).pretty_format(show_percents=True, values_in_chart=True, truncate=15, sort_by_count=True))

