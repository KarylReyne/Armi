import numpy as np
from nltk import word_tokenize, ngrams
from numpy.linalg import norm


def get_postag_vector(postag_vector, train_set, tagger):
    large_doc = []
    [large_doc.append(token) for tuple in train_set for token in word_tokenize(tuple[0])]
    for triple in tagger.tag_sent(large_doc):
        postag_vector[triple[2]] = 0
    dict(sorted(postag_vector.items()))


def preprocess(document, tagger):
    tokens = word_tokenize(document)
    # using the Hanover Tagger, not STTS
    # https://serwiss.bib.hs-hannover.de/frontdoor/index/index/docId/1527
    # (token, lemma, tag)
    document_triple = tagger.tag_sent(tokens)
    # TODO: dependency parsing
    return tokens, document_triple


def n_grams(features, tokens, n_list=[1, 2]):
    for n in n_list:
        for ngram in ngrams(tokens, n):
            features['contains {0}-gram {1}'.format(n, ngram)] = True


def l2norm_nparray(array):
    return array/norm(array) if norm(array) != 0 else array


def pos_tag_distribution(features, postag_vector, document_triple):
    # count postags
    for (token, lemma, tag) in document_triple:
        postag_vector[tag] += 1

    # get l2-normalized pos tag distribution vector
    array = np.zeros(len(postag_vector))
    i = 0
    for key in postag_vector:
        array[i] = postag_vector[key]
        postag_vector[key] = 0
        i += 1
    array = l2norm_nparray(array)

    j = 0
    for tag in postag_vector:
        if array[j] != 0:
            features["{0} count".format(tag)] = array[j]
        j += 1

    # # convert array to hashable string
    # pos_dist = "["
    # for j in range(0, i):
    #     pos_dist += "{0}, ".format(array[j])
    # pos_dist = pos_dist.rstrip(", ")+"]"
    #
    # features["POS tag distribution"] = pos_dist
    # print("POS tag distribution: {0}".format(features["POS tag distribution"]))


def structural_features(features, tokens):
    sum = len(tokens)
    features["token count"] = sum
    sum_c = 0
    sum_d = 0
    for t in tokens:
        sum_c += 1 if t == ',' else 0
        sum_d += 1 if t == '.' else 0
    features["% of comma tokens"] = sum_c / sum
    features["% of dot tokens"] = sum_d / sum
    features["last token"] = 'OTHER' if tokens[sum-1] not in ['.', '!', '?'] else tokens[sum-1]


def features(document, postag_vector, tagger):
    tokens, document_triple = preprocess(document, tagger)
    features = {}

    n_grams(features, tokens, n_list=[1])
    pos_tag_distribution(features, postag_vector, document_triple)
    structural_features(features, tokens)

    return features