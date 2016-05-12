import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
# from multiprocessing import cpu_count
from joblib import Parallel, delayed
from scipy.stats import binom_test, binom


# Temporarily stop warnings to cope with the too verbose sklearn
# GridSearchCV.score warning:
import warnings
warnings.simplefilter("ignore")


# boundaries for seeds generation during parallel processing:
MAX_INT = np.iinfo(np.uint32(1)).max
MIN_INT = np.iinfo(np.uint32(1)).min


def estimate_pvalue(score_unpermuted, scores_null):
    iterations = len(scores_null)
    p_value = max(1.0/iterations, (scores_null > score_unpermuted).sum() /
                  float(iterations))
    return p_value


def correct_predictions_scorer(estimator, X, y):
    """Count the number of correct predictions.
    """
    return np.float((estimator.predict(X) == y).sum())


def compute_clf_score_nestedCV(X, y, n_folds,
                               classifier=LogisticRegression,
                               classifier_param={},
                               scoring='accuracy',
                               random_state=None,
                               param_grid=[{'C': np.logspace(-5, 5, 20)}],
                               n_jobs=1):
    cv = StratifiedKFold(y, n_folds=n_folds, shuffle=True,
                         random_state=random_state)
    scores = np.zeros(n_folds)
    for i, (train, test) in enumerate(cv):
        cvclf = classifier(**classifier_param)
        y_train = y[train]
        cvcv = StratifiedKFold(y_train, n_folds=n_folds,
                               shuffle=True,
                               random_state=random_state)
        clf = GridSearchCV(cvclf, param_grid=param_grid, scoring=scoring,
                           cv=cvcv, n_jobs=n_jobs)
        clf.fit(X[train, :], y_train)
        scores[i] = clf.score(X[test, :], y[test])

    if scoring == correct_predictions_scorer:
        result = scores.sum()
    else:
        result = scores.mean()

    return result


def generate_data(muA, covA, muB, covB, nA, nB, rng_data):
    A = rng_data.multivariate_normal(muA, covA, size=nA)
    B = rng_data.multivariate_normal(muB, covB, size=nB)

    X = np.vstack([A, B])
    y = np.concatenate([np.zeros(nA), np.ones(nB)])
    return X, y


if __name__ == '__main__':

    np.random.seed(0)

    p_value_threshold = [0.05, 0.01]

    nA = 20  # size of class A
    nB = 20  # size of class B
    d = 200  # number of dimensions
    d_informative = 10  # number of informative dimensions

    # separation between the two normally-distributed classes:
    delta = 0.5

    print("nA = %s" % nA)
    print("nB = %s" % nB)
    print("d = %s" % d)
    print("d_informative = %s" % d_informative)
    print("delta = %s" % delta)

    muA = np.zeros(d)
    muB = np.concatenate([np.ones(d_informative) * delta,
                          np.zeros(d - d_informative)])
    print("muA: %s" % muA)
    print("muB: %s" % muB)
    covA = np.eye(d)
    covB = np.eye(d)
    # covB = np.diag(np.concatenate([np.ones(d_informative) *
    #                                np.random.uniform(low=0.0,
    #                                                  high=2.0,
    #                                                  size=d_informative),
    #                                np.ones(d - d_informative)]))
    print("covA: %s" % covA)
    print("covB: %s" % covB)

    test_methodology = 'binomial test'
    # test_methodology = 'permutation test'
    # test_methodology = 'confidence interval'
    print("Testing Methodology: %s" % test_methodology)

    scoring = correct_predictions_scorer  # 'accuracy'  #
    n_folds = 5
    if scoring == correct_predictions_scorer:
        scoring_name = 'num. correct predictions'
        chance_level = 0.5 * (nA + nB)
    else:
        scoring_name = scoring
        print
        chance_level = 0.5

    print("%s as test statistic." % scoring_name)

    iterations_unpermuted = 10
    iterations_permutations = 100

    seed_data = 0  # random generation of data
    rng_data = np.random.RandomState(seed_data)

    seed_cv = 0  # random splits of cross-validation
    rng_cv = np.random.RandomState(seed_cv)

    # clf = LogisticRegression
    # clf_params = {'penalty': 'l1'}
    # clf_param_grid = [{'C': np.logspace(-3, 3, 10)}]

    # clf = LogisticRegression
    # clf_params = {'penalty': 'l2'}
    # clf_param_grid = [{'C': np.logspace(-3, 3, 10)}]

    # clf = LogisticRegression
    # clf_params = {}
    # clf_param_grid = [{'C': np.logspace(-3, 3, 10)}]

    # clf = Perceptron
    # clf_params = {}
    # clf_param_grid = {}

    clf = SVC
    clf_params = {'kernel': 'linear'}
    clf_param_grid = [{'C': np.logspace(-3, 1, 10)}]

    # clf = SVC
    # clf_params = {'kernel': 'rbf'}
    # clf_param_grid = [{'C': np.logspace(-3, 3, 10)},
    #                   {'gamma': np.logspace(-5, 5, 10)}]

    print("Classifier: %s" % clf(**clf_params))

    nA_big = 300
    nB_big = 300
    X_big, y_big = generate_data(muA, covA, muB, covB, nA_big, nB_big,
                                 rng_data)
    score_big = compute_clf_score_nestedCV(X_big, y_big, n_folds, clf,
                                           clf_params,
                                           scoring, random_state=0,
                                           param_grid=clf_param_grid,
                                           n_jobs=-1)
    print("Asymptotic score of the classifier (%s examples): %s" %
          (nA_big + nB_big, score_big))

    repetitions = 100
    print("This experiments will be repeated on %s randomly-sampled datasets."
          % repetitions)

    scores = np.zeros(repetitions)
    p_value_scores = np.zeros(repetitions)
    for r in range(repetitions):
        print("")
        print("Repetition %s" % r)

        X, y = generate_data(muA, covA, muB, covB, nA, nB, rng_data)

        # score_unpermuted = compute_clf_score_nestedCV(X, y, n_folds,
        #                                               scoring=scoring,
        #                                               random_state=rng_cv)

        rngs = [np.random.RandomState(rng_cv.randint(low=MIN_INT, high=MAX_INT)) for i in range(iterations_unpermuted)]
        scores_unpermuted = Parallel(n_jobs=-1)(delayed(compute_clf_score_nestedCV)(X, y, n_folds, clf, clf_params, scoring, rngs[i], param_grid=clf_param_grid, n_jobs=-1) for i in range(iterations_unpermuted))
        scores_unpermuted = np.array(scores_unpermuted)
        if scoring == correct_predictions_scorer:
            # if score is integer, then use (rounded) median:
            score_unpermuted = np.median(scores_unpermuted).round()
        else:
            # otherwise use the mean:
            score_unpermuted = np.mean(scores_unpermuted)

        print("%s: %s" % (scoring_name, score_unpermuted))
        scores[r] = score_unpermuted

        if test_methodology == 'permutation test':
            # print("Doing permutations:"),
            scores_null = np.zeros(iterations_permutations)

            # for i in range(iterations_permutations):
            #     if (i % 10) == 0:
            #         print(i)

            #     yi = rng_cv.permutation(y)
            #     scores_null[i] = compute_clf_score_nestedCV(X, yi, n_folds,
            #                                                 scoring=scoring,
            #                                                 random_state=rng_cv)

            rngs = [np.random.RandomState(rng_cv.randint(low=MIN_INT, high=MAX_INT)) for i in range(iterations_permutations)]
            yis = [np.random.permutation(y) for i in range(iterations_permutations)]
            scores_null = Parallel(n_jobs=-1)(delayed(compute_clf_score_nestedCV)(X, yis[i], n_folds, clf, clf_params, scoring, rngs[i], param_grid=clf_param_grid) for i in range(iterations_permutations))
            p_value_score = estimate_pvalue(score_unpermuted, scores_null)

        elif test_methodology == 'binomial test':
            # two-sided:
            # p_value_score = binom_test(score_unpermuted, n=nA+nB, p=0.5)

            # one-sided:
            rv = binom(n=nA+nB, p=0.5)
            if scoring_name == 'accuracy':
                score_unpermuted = np.round(score_unpermuted * (nA + nB))

            p_value_score = rv.sf(score_unpermuted)  # sf: surv. funct.
        elif test_methodology == 'confidence interval':
            # Use the exact Binomial confidence interval for true
            # accuracy (Pereira 2009, Langford 2005) to compute how
            # much tail (as p-value) of the interval is below the
            # chance level.
            if scoring_name == 'accuracy':
                score_unpermuted = np.round(score_unpermuted * (nA + nB))
            else:
                # raise Exception  # the following works only for accuracy
                chance_level = 0.5

            p_grid = np.arange(0.0, 1.0, 0.01)
            inv_binom = np.array([binom(n=nA+nB, p=p).pmf(score_unpermuted)
                                  for p in p_grid])
            p_value_score = (inv_binom[p_grid <= chance_level]).sum() / inv_binom.sum()
        else:
            raise Exception

        p_value_scores[r] = p_value_score

        print("%s p-value: %s" % (scoring_name, p_value_score))

        for pvt in p_value_threshold:
            scores_power = (p_value_scores[:r+1] <= pvt).mean()
            print("p_value_threshold: %s" % pvt)
            print("Power (partial results) = %s" %
                  (scores_power,))

    print("")
    print("FINAL RESULTS:")
    for pvt in p_value_threshold:
        print("pvt: %s" % pvt)
        scores_power = (p_value_scores <= pvt).mean()
        print("%s Power: %s" % (scoring_name, scores_power))
        print("")
