import warnings

import pandas as pd
from aif360.algorithms.inprocessing import GridSearchReduction
# fairness tools
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from fairlearn.reductions import DemographicParity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# metrics
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from methods.regression_measures import compute_sp_avg_outcome, normalisation

from datasets_processing.aif360datset import get_aif_dataset

warnings.filterwarnings("ignore")


def funct_average_predictive_value_difference(classified_metric):
    return 0.5 * (classified_metric.difference(classified_metric.positive_predictive_value)
                  + classified_metric.difference(classified_metric.false_omission_rate))


# compute fairness metrics for two classes
def fair_metrics(dataset, y_predicted, privileged_groups, unprivileged_groups):
    dataset_pred = dataset.copy()

    dataset_pred.labels = y_predicted

    classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)

    print()

    result = {'statistical_parity_difference': metric_pred.statistical_parity_difference(),
              'disparate_impact': metric_pred.disparate_impact(),
              'equal_opportunity_difference': classified_metric.equal_opportunity_difference(),
              'average_odds_difference': classified_metric.average_odds_difference(),
              'average_predictive_value_difference': funct_average_predictive_value_difference(classified_metric),
              'false_discovery_rate_difference': classified_metric.false_discovery_rate_difference()}

    return result, classified_metric, metric_pred

# optimización ...
# Data complexity... para cada subgrupo.. Author Ho. https://pypi.org/project/data-complexity/



def apply_log_regression(dataset, splits=10, mitigation=False, rand_state=1):
    if dataset.outcome_type != 'binary':
        raise AssertionError('Only binary datasets allowed')

    lr_estimator = LogisticRegression(penalty='none', solver='lbfgs')

    min_v = min(dataset.ds[dataset.continuous_label_name])
    max_v = max(dataset.ds[dataset.continuous_label_name])

    accuracies = []
    statistical_parity_difference = []
    disparate_impact = []
    equal_opportunity_difference = []
    average_odds_difference = []
    average_predictive_value_difference = []
    mae = []
    mse = []
    mae_prob = []
    diff_err = []
    diff_err_prob = []
    acc_m = []
    false_discovery_rate_difference = []
    sp_avg_outcome_n = []
    sp_avg_outcome = []
    mae_n = []

    for num in range(0, 10):

        seed = 100+num
        train = pd.read_csv("/Users/jsuarez/Documents/Personal/fairness_in_regression/data/train_val_test_standard/{}/{}_output_{}_{}_train_seed_{}.csv".format(dataset._name, dataset._name, dataset.outcome_type, dataset._protected_att_name[0], seed))

        test_b = pd.read_csv("/Users/jsuarez/Documents/Personal/fairness_in_regression/data/train_val_test_standard/{}/{}_output_{}_{}_test_seed_{}.csv".format(dataset._name, dataset._name, dataset.outcome_type, dataset._protected_att_name[0], seed))
        test_o = pd.read_csv("/Users/jsuarez/Documents/Personal/fairness_in_regression/data/train_val_test_standard/{}/{}_output_continuous_test_seed_{}.csv".format(dataset._name, dataset._name, seed))


        train.rename(columns={'y': dataset.binary_label_name}, inplace=True)
        test_b.rename(columns={'y': dataset.binary_label_name}, inplace=True)
        test_o.rename(columns={'y': dataset.continuous_label_name}, inplace=True)


        X_train = train.drop([dataset.binary_label_name], axis=1)
        X_test = test_b.drop([dataset.binary_label_name], axis=1)

        y_train = train[dataset.binary_label_name]

        y_test_binary = test_b[dataset.binary_label_name]
        y_test_ordinal = test_o[dataset.continuous_label_name]
        y_test_norm = normalisation(y_test_ordinal, min_v, max_v)
        y_test_ordinal_normalized = pd.DataFrame(y_test_norm, columns=[dataset.continuous_label_name])



        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test_binary.reset_index(drop=True, inplace=True)
        y_test_ordinal.reset_index(drop=True, inplace=True)
        y_test_ordinal_normalized.reset_index(drop=True, inplace=True)


        column_predicted = dataset.outcome_label + '_predicted'
        target_variable = dataset.outcome_label  # if binary else target_variable_ordinal

        y_test = y_test_binary
        favorable_label = dataset.favorable_label_binary

        if not mitigation:
            clf = lr_estimator.fit(X_train, y_train)
            results_ = clf.predict(X_test)
            results = pd.DataFrame(results_, columns=[column_predicted])

            # transforming the output to ordinal values
            # NOTE esto se hace para calcular MSE con la salida ordinal o continua.
            probabilities = clf.predict_proba(X_test)
            prob_df = pd.DataFrame(probabilities[:, 1], columns=['prob_class_1'])
            if dataset._outcome_original == 'ordinal':
                ordinal_output = dataset.compute_ordinal_output(results_, probabilities, False)
                ordinal_output_prob = dataset.compute_ordinal_output(results_, probabilities, True)
                mae_prob.append(mean_absolute_error(y_test_ordinal, ordinal_output_prob))
                # print(y_test_ordinal[dataset.continuous_label_name].tolist())
                diff_err_prob.append(dataset.compute_diff_error(y_test_ordinal,
                                                                ordinal_output_prob))
            else:
                ordinal_output = dataset.compute_continuous_output(results_, probabilities[:, 1])

            diff_err.append(
                dataset.compute_diff_error(y_test_ordinal, ordinal_output))
            mae.append(mean_absolute_error(y_test_ordinal, ordinal_output))



        else:
            # LSAT Generating a grid with 10 grid points. It is recommended to use at least 16 grid points. Please consider increasing grid_size.
            grid_search_red = GridSearchReduction(prot_attr=dataset.protected_att_name,
                                                  estimator=lr_estimator,
                                                  constraints=DemographicParity(),
                                                  loss="Absolute",
                                                  min_val=0,
                                                  max_val=1,
                                                  grid_size=16,  # TODO this is 10 by default
                                                  drop_prot_attr=False)

            target_variable = dataset.binary_label_name
            y_test_ = y_test_binary

            ds_tra = get_aif_dataset(X_train, y_train, label=target_variable,
                                     protected_attribute_names=dataset.protected_att_name,
                                     privileged_classes=dataset.privileged_classes,
                                     favorable_classes=favorable_label)
            ds_tst = get_aif_dataset(X_test, y_test_, label=target_variable,
                                     protected_attribute_names=dataset.protected_att_name,
                                     privileged_classes=dataset.privileged_classes,
                                     favorable_classes=favorable_label)
            grid_search_red.fit(ds_tra)
            gs_pred = grid_search_red.predict(ds_tst)
            results = pd.DataFrame(gs_pred.labels, columns=[column_predicted])

            # transforming the output to ordinal values
            if dataset._outcome_original == 'ordinal':
                probabilities = []
                ordinal_output = dataset.compute_ordinal_output(gs_pred.labels, probabilities, False)
            else:
                ordinal_output = dataset.compute_continuous_output(gs_pred.labels, probabilities)
            mae.append(mean_absolute_error(y_test_ordinal, ordinal_output))
            diff_err.append(
                dataset.compute_diff_error(y_test_ordinal[dataset.continuous_label_name].tolist(), ordinal_output))

        results_normalized = normalisation(ordinal_output, min_v, max_v)
        results_n = pd.DataFrame(results_normalized, columns=[column_predicted])
        sp_avg_outcome.append(
            compute_sp_avg_outcome(ordinal_output, X_test[dataset.protected_att_name[0]].values.tolist(),
                                   dataset.privileged_classes[0][0]))
        sp_avg_outcome_n.append(
            compute_sp_avg_outcome(results_normalized, X_test[dataset.protected_att_name[0]].values.tolist(),
                                   dataset.privileged_classes[0][0]))
        mae_n.append(mean_absolute_error(y_test_ordinal_normalized[dataset.continuous_label_name], results_normalized))
        mse.append(mean_squared_error(y_test_ordinal_normalized[dataset.continuous_label_name], results_normalized))

        results_cm = cm(y_test, results)
        print(results_cm)
        acc = accuracy_score(y_test, results)
        print(acc)
        accuracies.append(acc)

        ds_tra = get_aif_dataset(X_test, y_test, label=target_variable,
                                 protected_attribute_names=dataset.protected_att_name,
                                 privileged_classes=dataset.privileged_classes,
                                 favorable_classes=favorable_label)

        res, classm, predm = fair_metrics(ds_tra, results[column_predicted], dataset.privileged_groups,
                                          dataset.unprivileged_groups)
        statistical_parity_difference.append(predm.statistical_parity_difference())
        disparate_impact.append(predm.disparate_impact())
        equal_opportunity_difference.append(classm.equal_opportunity_difference())
        average_odds_difference.append(classm.average_odds_difference())
        average_predictive_value_difference.append(funct_average_predictive_value_difference(classm))
        false_discovery_rate_difference.append(classm.false_discovery_rate_difference())
        acc_m.append(classm.accuracy())
        print(res)


    if mitigation or dataset._outcome_original != 'ordinal':
        mae_prob = mae
        diff_err_prob = diff_err

    dict_metrics = {'accuracies': accuracies,
                    'acc_m': acc_m,
                    'statistical_parity_difference': statistical_parity_difference,
                    'disparate_impact': disparate_impact,
                    'equal_opportunity_difference': equal_opportunity_difference,
                    'average_odds_difference': average_odds_difference,
                    'average_predictive_value_difference': average_predictive_value_difference,
                    'false_discovery_rate_difference': false_discovery_rate_difference,
                    'mean_absolute_error': mae,
                    'mae': mae,
                    'mean_absolute_error_prob': mae_prob,
                    'diff_err': diff_err,
                    'diff_err_prob': diff_err_prob,
                    'sp_avg_outcome_n': sp_avg_outcome_n,
                    'sp_avg_outcome': sp_avg_outcome,
                    'mse_n': mse,
                    'mae_n': mae_n
                    }
    df_metrics = pd.DataFrame(dict_metrics)
    print(df_metrics)
    return df_metrics
