import de_core_news_sm
import xmltodict
from nltk import NaiveBayesClassifier, classify
from argminer_features import get_xtag_vector
from processing.thread_manager import ThreadManager
from nltk.metrics import ConfusionMatrix


def ensure_spacy_dataset_presence(dataset):
    try:
        spcy = de_core_news_sm.load()
    except Exception as e:
        print(e)
        print("spacy dataset '{0}' not present. Install it via 'python -m spacy download {0}'".format(dataset))
        exit()


def read_corpora(task, collapse_claims=True):
    def get_lbl(lbl):
        if collapse_claims and (lbl == 'ClaimPro' or lbl == 'ClaimContra'):
            return 'Claim'
        return lbl

    train_file = "THF_corpus_v1.0/3.corpus/subtask{0}_train.xml".format(task)
    test_file = "THF_corpus_v1.0/3.corpus/subtask{0}_test.xml".format(task)

    with open(train_file, 'r', encoding="utf-8") as xml:
        string = xml.read()
        tmp = xmltodict.parse(string)
        train = []
        [train.append([dict['Text'], get_lbl(dict['Label'])]) for dict in tmp["root"]["Sentence"]]
    with open(test_file, 'r', encoding="utf-8") as xml:
        string = xml.read()
        tmp = xmltodict.parse(string)
        test = []
        [test.append([dict['Text'], get_lbl(dict['Label'])]) for dict in tmp["root"]["Sentence"]]
    return train, test


def map_feature_extraction(extraction_function, task):
    train_corpus, test_corpus = read_corpora(task)

    # spcy = spacy.load("de_core_news_sm")
    spcy = de_core_news_sm.load()

    tm = ThreadManager()

    postag_vector = {}
    deptag_vector = {}
    tm.execute_as_new_process(id="get_postag_vector",
                              target=get_xtag_vector,
                              args=(postag_vector, "pos", train_corpus, spcy,))
    tm.execute_as_new_process(id="get_deptag_vector",
                              target=get_xtag_vector,
                              args=(deptag_vector, "dep", train_corpus, spcy,))
    train_set = []
    tm.execute_as_new_process(id="get_train_set",
                              target=(lambda tc, pv, dv, s:
                                      [train_set.append((extraction_function(d, pv, dv, s), a)) for (d, a) in tc]),
                              args=(train_corpus, postag_vector, deptag_vector, spcy,))
    test_set = []
    tm.execute_as_new_process(id="get_test_set",
                              target=(lambda tc, pv, dv, s:
                                      [test_set.append((extraction_function(d, pv, dv, s), a)) for (d, a) in tc]),
                              args=(test_corpus, postag_vector, deptag_vector, spcy,))

    return test_set, train_set


def evaluate_feature_extraction(_extraction_function, _classifier=None, task="A", header='\nNaive Bayes Classifier'):
    def shorten_label(string):
        short = string
        if string == 'MajorPosition': short = 'MP'
        elif string == "Premise": short = 'P'
        elif string == 'Claim': short = 'C'
        elif string == 'argumentative': short = "A"
        elif string == 'non-argumentative': short = "nA"
        return short

    test_set, train_set = map_feature_extraction(extraction_function=_extraction_function, task=task)

    classifier = NaiveBayesClassifier.train(train_set) if _classifier is None else _classifier(train_set)
    accuracy = classify.accuracy(classifier, test_set)
    print(header)
    print("Accuracy: ", accuracy)
    classifier.show_most_informative_features(20) if _classifier is None else print(end='')
    gold = [shorten_label(l) for (f, l) in test_set]
    test = [shorten_label(classifier.classify(f)) for (f, l) in test_set]
    print(ConfusionMatrix(gold, test).pretty_format(show_percents=True, values_in_chart=True,
                                                    truncate=15, sort_by_count=True))

