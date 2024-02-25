import warnings

import pandas as pd
# fairness tools
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.metrics import accuracy_score
# metrics
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import mean_absolute_error

from datasets_processing.aif360datset import get_aif_dataset

from sklearn.tree import DecisionTreeRegressor, plot_tree
from methods.regression_measures import compute_sp_ks, compute_sp_avg_outcome, compute_ea, compute_metrics, normalisation

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

    result = {'statistical_parity_difference': metric_pred.statistical_parity_difference(),
              'disparate_impact': metric_pred.disparate_impact(),
              'equal_opportunity_difference': classified_metric.equal_opportunity_difference(),
              'average_odds_difference': classified_metric.average_odds_difference(),
              'average_predictive_value_difference': funct_average_predictive_value_difference(classified_metric),
              'false_discovery_rate_difference': classified_metric.false_discovery_rate_difference()}

    return result, classified_metric, metric_pred

# optimización ...
# Data complexity... para cada subgrupo.. Author Ho. https://pypi.org/project/data-complexity/


def apply_c45_regressor(dataset, splits=10, mitigation=False, rand_state=1):
    if dataset.outcome_type == 'binary':
        raise AssertionError('Only non binary datasets allowed')

    lr_estimator = DecisionTreeRegressor(random_state=42)
    # lr_estimator = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)  # With cost complexity pruning ccp_alpha=0.01,

    y_ord = dataset.ds[dataset.continuous_label_name]
    min_v = min(y_ord)
    max_v = max(y_ord)

    accuracies = []
    statistical_parity_difference = []
    disparate_impact = []
    equal_opportunity_difference = []
    average_odds_difference = []
    average_predictive_value_difference = []
    mae = []
    mae_n = []
    mae_prob = []
    diff_err = []
    diff_err_prob = []
    acc_m = []
    false_discovery_rate_difference = []
    sp_mi_ = []
    sp_ks_ = []
    sp_avg_outcome_ = []
    ea_ = []
    metrics_ = []
    didi_ = []
    didi_predicted_ = []
    sp_mi_n = []
    sp_ks_n = []
    sp_avg_outcome_n = []
    ea_n = []
    metrics_n = []
    didi_n = []
    didi_predicted_n = []
    deo = []

    for num in range(0, 10):

        seed = 100 + num

        df_tra = pd.read_csv(
            "/Users/juls/Desktop/BIAS- regression/compas_analysis/data/train_val_test/{}/{}_output_continuous_test_seed_{}.csv".format(
                dataset._name, dataset._name, seed))

        df_tst_c = pd.read_csv(
            "/Users/juls/Desktop/BIAS- regression/compas_analysis/data/train_val_test/{}/{}_output_continuous_test_seed_{}.csv".format(
                dataset._name, dataset._name, seed))
        df_tst_b = pd.read_csv(
            "/Users/juls/Desktop/BIAS- regression/compas_analysis/data/train_val_test/{}/{}_output_binary_test_seed_{}.csv".format(
                dataset._name, dataset._name, seed))

        df_tra.rename(columns={'y': dataset.continuous_label_name}, inplace=True)
        df_tst_b.rename(columns={'y': dataset.binary_label_name}, inplace=True)
        df_tst_c.rename(columns={'y': dataset.continuous_label_name}, inplace=True)

        X_train = df_tra[dataset.explanatory_variables]
        X_test = df_tst_c[dataset.explanatory_variables]
        y_train = df_tra[dataset.continuous_label_name]
        y_test_binary = df_tst_b[dataset.binary_label_name]
        y_test_ordinal = df_tst_c[dataset.continuous_label_name]

        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test_binary.reset_index(drop=True, inplace=True)
        y_test_ordinal.reset_index(drop=True, inplace=True)

        column_predicted = dataset.outcome_label + '_predicted'
        target_variable = dataset.outcome_label  # if binary else target_variable_ordinal

        y_test = y_test_binary
        favorable_label = dataset.favorable_label_binary

        clf = lr_estimator.fit(X_train, y_train)
        results_ = clf.predict(X_test)
        results = pd.DataFrame(results_, columns=[column_predicted])



        metrics_.append(compute_metrics(y_test, results[column_predicted]))

        sp_ks_.append(compute_sp_ks(results[column_predicted], X_test[dataset.protected_att_name[0]].values.tolist(),
                                    dataset.privileged_classes[0][0]))
        sp_avg_outcome_.append(
            compute_sp_avg_outcome(results[column_predicted], X_test[dataset.protected_att_name[0]].values.tolist(),
                                   dataset.privileged_classes[0][0]))


        ea_.append(compute_ea(y_test, results[column_predicted],  # [dataset.continuous_label_name]
                              X_test[dataset.protected_att_name[0]].values.tolist(), dataset.privileged_classes[0][0]))

        normalised_results = normalisation(results[column_predicted], min_v, max_v)
        sp_avg_outcome_n.append(
            compute_sp_avg_outcome(normalised_results, X_test[dataset.protected_att_name[0]].values.tolist(),
                                   dataset.privileged_classes[0][0]))
        mae.append(mean_absolute_error(y_test, results[column_predicted]))
        y_test_normalised = normalisation(y_test, min_v, max_v)
        mae_n.append(mean_absolute_error(y_test_normalised, normalised_results))

        # transforming the output to binary values
        # y_test_binary
        y_predicted_binary = dataset.continuous_to_binary(results[column_predicted])


        results_cm = cm(y_test_binary, y_predicted_binary)
        print(results_cm)
        acc = accuracy_score(y_test_binary, y_predicted_binary)
        print(acc)
        accuracies.append(acc)

        ds_tra = get_aif_dataset(X_test, y_test_binary, label=dataset.binary_label_name,
                                 protected_attribute_names=dataset.protected_att_name,
                                 privileged_classes=dataset.privileged_classes,
                                 favorable_classes=favorable_label)


        res, classm, predm = fair_metrics(ds_tra, y_predicted_binary, dataset.privileged_groups,
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
                    'sp_ks': sp_ks_,
                    'sp_avg_outcome': sp_avg_outcome_,
                    'ea': ea_,
                    'sp_avg_outcome_n': sp_avg_outcome_n,
                    'mae': mae,
                    'mae_n': mae_n
                    }
    df_metrics = pd.DataFrame(dict_metrics)
    print(df_metrics)
    return df_metrics
