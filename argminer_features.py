import numpy as np
from nltk import ngrams
from numpy.linalg import norm


def get_xtag_vector(xtag_vector, tag_str, train_set, spacy):
    large_doc = []
    [large_doc.append(tuple[0]) for tuple in train_set]
    large_doc = " ".join(large_doc)
    for item in spacy(large_doc):
        tag = item.pos_ if tag_str == "pos" else item.dep_
        xtag_vector[tag] = 0
    dict(sorted(xtag_vector.items()))


def preprocess(document, spacy):
    annotated_doc = spacy(document)
    tok = []
    pos = []
    dep = []
    lem = []
    [tok.append(item.text) for item in annotated_doc]
    [pos.append(item.pos_) for item in annotated_doc]
    [dep.append(item.dep_) for item in annotated_doc]
    [lem.append(item.lemma_) for item in annotated_doc]
    return tok, pos, dep, lem


def n_grams(features, tokens, n_list=[1, 2]):
    for n in n_list:
        lowercase_tokens = set()
        [lowercase_tokens.add(token.lower()) for token in tokens]
        for ngram in ngrams(lowercase_tokens, n):
            features['contains {0}-gram {1}'.format(n, ngram)] = True


def l2norm_nparray(array):
    return array/norm(array) if norm(array) != 0 else array


def xtag_distribution(features, tag_vector, doc_tags):
    # count postags
    for tag in doc_tags:
        try:
            tag_vector[tag] += 1
        except KeyError as e:
            #print("{0} tag missing in tag_vector".format(e))
            tag_vector[tag] = 1

    # get l2-normalized pos tag distribution vector
    array = np.zeros(len(tag_vector))
    i = 0
    for key in tag_vector:
        array[i] = tag_vector[key]
        tag_vector[key] = 0
        i += 1
    array = l2norm_nparray(array)

    j = 0
    for tag in tag_vector:
        if array[j] != 0:
            features["{0} count".format(tag)] = array[j]
        j += 1


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


def features(document, postag_vector, deptag_vector, spacy):
    tok, pos, dep, lem = preprocess(document, spacy)
    features = {}

    n_grams(features, tok, n_list=[1])
    xtag_distribution(features, postag_vector, pos)
    xtag_distribution(features, deptag_vector, dep)
    structural_features(features, tok)

    return features