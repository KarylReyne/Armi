import cProfile
from nltk import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from argminer_features import features
from argminer_util import evaluate_feature_extraction, check_spacy_dataset_presence
from processing.thread_manager import ThreadManager


feature_function = (lambda tc, pv, dv, s: features(tc, pv, dv, s))


def get_stats(string, file="restats", n=10):
    import pstats
    from pstats import SortKey
    cProfile.run(string, file)
    p = pstats.Stats(file)
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(n)


def nbc(task):
    evaluate_feature_extraction(feature_function,
                                task=task,
                                header='\nNaive Bayes Classifier (task {0})'.format(task))


def knn(task):
    evaluate_feature_extraction(feature_function,
                                (lambda x: SklearnClassifier(KNeighborsClassifier()).train(x)),
                                task=task,
                                header="\nkNN Classifier (task {0})".format(task))


def svm(task, gs=False):
    param_grid = {'kernel': ('linear', 'rbf', 'poly'),
                  'C': [1, 10, 100],
                  'gamma': ['auto', 'scale']
                  }
    estimator = GridSearchCV(SVC(), param_grid, cv=10) if gs else SVC(kernel='linear')
    evaluate_feature_extraction(feature_function,
                                (lambda x: SklearnClassifier(estimator).train(x)),
                                task=task,
                                header="\nSupport Vector Classifier (task {0})".format(task))
    print("best_params: {0}".format(estimator.best_params_)) if gs else print(end='')


def rfc(task):
    evaluate_feature_extraction(feature_function,
                                (lambda x: SklearnClassifier(RandomForestClassifier()).train(x)),
                                task=task,
                                header="\nRandom Forest Classifier (task {0})".format(task))


if __name__ == '__main__':
    check_spacy_dataset_presence()

    compute_in_sequence = True

    tm = ThreadManager()

    for task in ["A", "B"]:
        tm.execute_as_new_process(id='NBC {0}'.format(task), target=nbc, args=(task,), join=compute_in_sequence)
        tm.execute_as_new_process(id='KNN {0}'.format(task), target=knn, args=(task,), join=compute_in_sequence)
        tm.execute_as_new_process(id='SVM {0}'.format(task), target=svm, args=(task, False,), join=compute_in_sequence)
        tm.execute_as_new_process(id='RFC {0}'.format(task), target=rfc, args=(task,), join=compute_in_sequence)

    # task = "B"
    # get_stats("nbc(task)", n=10)
    # get_stats("knn(task)", n=10)
    # get_stats("svm(task)", n=10)
    # get_stats("rfc(task)", n=10)
