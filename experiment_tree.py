import os
import sys
import warnings
import numpy as np
import pandas as pd

from datasets_processing.compas import CompasDataset
from datasets_processing.datasets_utils import get_datasets
from methods.c45_classification import apply_c45_classifier
from methods.c45_regression import apply_c45_regressor


warnings.filterwarnings("ignore")
to_insert = os.getcwd()
sys.path.append(to_insert)

def run_experiment(dataset, mitigation, outcome, rand_state):
    if dataset.name == 'older-adults':
        if outcome == 'binary':
            return apply_c45_classifier(dataset, splits=3, mitigation=mitigation, rand_state=rand_state)
        elif outcome == 'ordinal':
            return apply_c45_regressor(dataset, splits=3, mitigation=mitigation)
        elif outcome == 'continuous':
            return apply_c45_regressor(dataset, splits=3, mitigation=mitigation)
        else:
            raise AssertionError('not a valid outcome: ', outcome)
    else:
        if outcome == 'binary':
            return apply_c45_classifier(dataset, splits=10, mitigation=mitigation, rand_state=rand_state)
        elif outcome == 'ordinal':
            return apply_c45_regressor(dataset, splits=10, mitigation=mitigation)
        elif outcome == 'continuous':
            return apply_c45_regressor(dataset, splits=10, mitigation=mitigation)
        else:
            raise AssertionError('not a valid outcome: ', outcome)


outcomes = ['continuous']  # , 'ordinal', 'binary'
mitigation = [False]
stats_results = True

for outcome in outcomes:
    for mit in mitigation:
        datasets = get_datasets(outcome)
        # datasets = [CompasDataset('race', outcome_type=outcome)]
        # datasets = [WineDataset('color', outcome_type=outcome)]
        for dataset in datasets:
            print(dataset, outcome, mit)
            appended_results = []
            rand_state = 42
            print('Random state: ', rand_state)
            results = run_experiment(dataset, mit, outcome, rand_state)
            results.replace([np.inf, -np.inf], np.nan, inplace=True)
            results['seed'] = rand_state
            appended_results.append(results)
            # Concatenate all results and print them
            appended_data = pd.concat(appended_results)
            if stats_results:
                # Print quartiles
                stats = appended_data.describe()
                stats.to_csv('results/regression_tree/{name}_{att}_{outcome}_{mitigation}_quartiles.csv'.format(name=dataset.name,
                                                                                                                outcome=outcome,
                                                                                                                mitigation=mit,
                                                                                                                att=dataset._att))


            appended_data.to_csv('results/regression_tree/{name}_{att}_{outcome}_{mitigation}.csv'.format(name=dataset.name,
                                                                                                          outcome=outcome,
                                                                                                          mitigation=mit,
                                                                                                          rand=rand_state,
                                                                                                          att=dataset._att),
                                 index=False)
