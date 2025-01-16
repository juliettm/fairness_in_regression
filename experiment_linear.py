import os
import sys
import warnings

from datasets_processing.datasets_utils import get_datasets
from methods.linear_regression import apply_lin_regression
from methods.logistic_regression import apply_log_regression
from methods.ordinal_regression import apply_ord_regression

warnings.filterwarnings("ignore")
to_insert = os.getcwd()
sys.path.append(to_insert)


def run_experiment(dataset, mitigation, outcome, rand_state):
    if dataset.name == 'older-adults':
        if outcome == 'binary':
            return apply_log_regression(dataset, splits=3, mitigation=mitigation, rand_state=rand_state)
        elif outcome == 'ordinal':
            return apply_ord_regression(dataset, splits=3, mitigation=mitigation)
        elif outcome == 'continuous':
            return apply_lin_regression(dataset, splits=3, mitigation=mitigation)
        else:
            raise AssertionError('not a valid outcome: ', outcome)
    else:
        if outcome == 'binary':
            return apply_log_regression(dataset, splits=10, mitigation=mitigation, rand_state=rand_state)
        elif outcome == 'ordinal':
            return apply_ord_regression(dataset, splits=10, mitigation=mitigation)
        elif outcome == 'continuous':
            return apply_lin_regression(dataset, splits=10, mitigation=mitigation)
        else:
            raise AssertionError('not a valid outcome: ', outcome)


outcomes = ['continuous']  # , 'ordinal', 'binary', 'continuous'
mitigation = [False]
stats_results = True

for outcome in outcomes:
    for mit in mitigation:
        datasets = get_datasets(outcome)
        for dataset in datasets:
            print(dataset, outcome, mit)
            appended_results = []
            rand_state = 42
            results = run_experiment(dataset, mit, outcome, rand_state)
            if stats_results:
                # Print the quartiles
                stats = results.describe()
                stats.to_csv('results/regression_linear/{name}_{outcome}_{mitigation}_{att}_quartiles.csv'.format(name=dataset.name,
                                                                                                                  outcome=outcome,
                                                                                                                  mitigation=mit,
                                                                                                                  att=dataset._att))

            results.to_csv('results/regression_linear/{name}_{outcome}_{mitigation}_{att}.csv'.format(name=dataset.name,
                                                                                                      outcome=outcome,
                                                                                                      mitigation=mit,
                                                                                                      rand=rand_state,
                                                                                                      att=dataset._att),
                           index=False)
