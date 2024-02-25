import warnings

import pandas as pd
# fairness tools
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.metrics import accuracy_score
# metrics
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import mean_absolute_error
from datasets_processing.aif360datset import get_aif_dataset
from sklearn.tree import DecisionTreeClassifier
from methods.regression_measures import compute_sp_avg_outcome, normalisation

warnings.filterwarnings("ignore")

def convert_to_continuous(binary_output, prob_class_1, min_val, max_val, threshold):
    """Convert a binary output to a continuous output using probabilities and scale it to min_val-max_val."""
    # return min_val + (max_val - min_val) * (binary_output * prob_class_1 + (1 - binary_output) * (1 - prob_class_1))
    def scale_0_to_1(original_value):
        return (original_value - 0.5) / 0.5

    if binary_output == 0:
        min_val = min_val
        max_val = threshold - 1
        prob = scale_0_to_1(1 - prob_class_1)
        result = max_val - (max_val - min_val) * prob if prob_class_1 < 0.5 else threshold - 1
    else:
        min_val = threshold
        max_val = max_val
        prob = scale_0_to_1(prob_class_1)
        result = min_val + (max_val - min_val) * prob if prob_class_1 > 0.5 else threshold
    return result



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



def apply_c45_classifier(dataset, splits=10, mitigation=False, rand_state=1):
    if dataset.outcome_type != 'binary':
        raise AssertionError('Only binary datasets allowed')

    min_output = dataset.ds[dataset._continuous_label_name].min()
    max_output = dataset.ds[dataset._continuous_label_name].max()
    threshold = dataset._cut_point


    lr_estimator = DecisionTreeClassifier(criterion='entropy', random_state=42)
    # lr_estimator = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)  # With cost complexity pruning ccp_alpha=0.01,

    accuracies = []
    statistical_parity_difference = []
    disparate_impact = []
    equal_opportunity_difference = []
    average_odds_difference = []
    average_predictive_value_difference = []
    mae = []
    mae_prob = []
    diff_err = []
    diff_err_prob = []
    acc_m = []
    false_discovery_rate_difference = []
    sp_avg_outcome_n = []
    mae_n = []


    for num in range(0, 10):

        seed = 100 + num

        df_tra = pd.read_csv(
            "/Users/juls/Desktop/BIAS- regression/compas_analysis/data/train_val_test/{}/{}_output_{}_train_seed_{}.csv".format(
                dataset._name, dataset._name, dataset.outcome_type, seed))

        # df_tra_ = pd.read_csv( "/Users/juls/Desktop/BIAS- regression/compas_analysis/data/train_val_test/{}/{}_output_continuous_train_seed_{}.csv".format( dataset._name, dataset._name, seed))

        df_tst_b = pd.read_csv(
            "/Users/juls/Desktop/BIAS- regression/compas_analysis/data/train_val_test/{}/{}_output_{}_test_seed_{}.csv".format(
                dataset._name, dataset._name, dataset.outcome_type, seed))
        df_tst_c = pd.read_csv(
            "/Users/juls/Desktop/BIAS- regression/compas_analysis/data/train_val_test/{}/{}_output_continuous_test_seed_{}.csv".format(
                dataset._name, dataset._name, seed))

        df_tra.rename(columns={'y': dataset.binary_label_name}, inplace=True)
        df_tst_b.rename(columns={'y': dataset.binary_label_name}, inplace=True)
        df_tst_c.rename(columns={'y': dataset.continuous_label_name}, inplace=True)

        X_train = df_tra[dataset.explanatory_variables]
        X_test = df_tst_c[dataset.explanatory_variables]
        y_train = df_tra[dataset.binary_label_name]
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
        pred_prob = clf.predict_proba(X_test)
        results_cont = pd.DataFrame(results_, columns=[column_predicted])
        #  Convert to continuous
        results_cont['prob_class_1'] = pred_prob[:, 1]
        results_cont['pred_cont'] = results_cont.apply(
            lambda x: convert_to_continuous(x[column_predicted], x['prob_class_1'], min_output, max_output, threshold), axis=1)

        normalised_results = normalisation(results[column_predicted], min_output, max_output)
        sp_avg_outcome_n.append(
            compute_sp_avg_outcome(normalised_results, X_test[dataset.protected_att_name[0]].values.tolist(),
                                   dataset.privileged_classes[0][0]))
        y_test_normalised = normalisation(y_test, min_output, max_output)
        mae_n.append(mean_absolute_error(y_test_normalised, normalised_results))

        results_cm = cm(y_test, results)
        print(results_cm)

        # Extract True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN)
        TP = results_cm[1, 1]
        TN = results_cm[0, 0]
        FP = results_cm[0, 1]
        FN = results_cm[1, 0]

        # Compute conditions
        is_FN_greater_than_TN = FN > TN
        is_FP_greater_than_TP = FP > TP

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
                    'FN-TN': int(is_FN_greater_than_TN),
                    'FP-TP': int(is_FP_greater_than_TP),
                    'sp_avg_outcome_n': sp_avg_outcome_n,
                    'mae_n': mae_n
                    }
    df_metrics = pd.DataFrame(dict_metrics)
    print(df_metrics)
    return df_metrics
