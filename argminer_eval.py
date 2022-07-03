import cProfile
from nltk import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from argminer_features import features
from argminer_util import evaluate_feature_extraction
from processing.thread_manager import ThreadManager


def get_stats(string, file="restats", n=10):
    import pstats
    from pstats import SortKey
    cProfile.run(string, file)
    p = pstats.Stats(file)
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(n)


def nbc():
    evaluate_feature_extraction((lambda x, y, z: features(x, y, z)))


def knn():
    evaluate_feature_extraction((lambda x, y, z: features(x, y, z)),
                                (lambda x: SklearnClassifier(KNeighborsClassifier()).train(x)),
                                header="\nkNN Classifier")


def svm(kernel="linear"):
    evaluate_feature_extraction((lambda x, y, z: features(x, y, z)),
                                (lambda x: SklearnClassifier(SVC(kernel=kernel)).train(x)),
                                header="\nSupport Vector Classifier with {0} kernel".format(kernel))


def rfc():
    evaluate_feature_extraction((lambda x, y, z: features(x, y, z)),
                                (lambda x: SklearnClassifier(RandomForestClassifier()).train(x)),
                                header="\nRandom Forest Classifier")


if __name__ == '__main__':
    tm = ThreadManager()
    #tm.execute_as_new_process(id='NBC', target=nbc, args=(), join=False)
    #tm.execute_as_new_process(id='KNN', target=knn, args=(), join=False)
    for kernel in [#"linear",
                   #"poly",
                   "rbf",
                   #"sigmoid"
                   ]:
        tm.execute_as_new_process(id='SVM', target=svm, args=(kernel,), join=False)
    #tm.execute_as_new_process(id='RFC', target=rfc, args=(), join=False)

    # get_stats("nbc()", n=10)
    # get_stats("svm()", n=10)
    # get_stats("dtc()", n=10)
