import warnings

import pandas as pd
# fairness tools
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from fairlearn.reductions import BoundedGroupLoss, ZeroOneLoss
from fairlearn.reductions import GridSearch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
# metrics
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import mean_absolute_error
# training

from datasets_processing.aif360datset import get_aif_dataset
from methods.regression_measures import sp_mi, compute_sp_avg_outcome, compute_metrics, normalisation

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


def apply_lin_regression(dataset, splits=10, mitigation=False):
    if dataset.outcome_type != 'continuous':
        raise AssertionError('Only continuous datasets allowed')

    lr_estimator = LinearRegression()

    min_v = min(dataset.ds[dataset.continuous_label_name])
    max_v = max(dataset.ds[dataset.continuous_label_name])
    #y_norm = normalisation(y_ordinal[dataset.continuous_label_name], min_v, max_v)
    #y_normalized = pd.DataFrame(y_norm, columns=[dataset.continuous_label_name])

    index_mi = dataset.explanatory_variables.index(dataset.protected_att_name[0])
    # print(index_mi)

    # y_ordinal = pd.DataFrame(scaler.fit_transform(dataset.ds[[dataset.continuous_label_name]]), columns=[dataset.continuous_label_name])
    # cut_point = scaler.transform(pd.DataFrame([dataset.get_cut_point()],columns=[dataset.continuous_label_name]))
    # dataset.set_cut_point(cut_point[0][0])
    a_name = dataset.protected_att_name

    accuracies = []
    statistical_parity_difference = []
    disparate_impact = []
    equal_opportunity_difference = []
    average_odds_difference = []
    average_predictive_value_difference = []
    false_discovery_rate_difference = []
    mae = []
    mae_n = []
    mae_prob = []
    diff_err = []
    diff_err_prob = []
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
    deo_ = []

    #y = y_ordinal

    for num in range(0, 10):

        seed = 100 + num
        df_tra = pd.read_csv(
            "/Users/juls/Desktop/BIAS- regression/compas_analysis/data/train_val_test_standard/{}/{}_output_continuous_train_seed_{}.csv".format(
                dataset._name, dataset._name, seed))
        df_tst_c = pd.read_csv(
            "/Users/juls/Desktop/BIAS- regression/compas_analysis/data/train_val_test_standard/{}/{}_output_continuous_test_seed_{}.csv".format(
                dataset._name, dataset._name, seed))
        df_tst_b = pd.read_csv(
            "/Users/juls/Desktop/BIAS- regression/compas_analysis/data/train_val_test_standard/{}/{}_output_binary_test_seed_{}.csv".format(
                dataset._name, dataset._name, seed))

        df_tra.rename(columns={'y': dataset.continuous_label_name}, inplace=True)
        df_tst_c.rename(columns={'y': dataset.continuous_label_name}, inplace=True)
        df_tst_b.rename(columns={'y': dataset.binary_label_name}, inplace=True)

        X_train = df_tra[dataset.explanatory_variables]
        X_test = df_tst_c[dataset.explanatory_variables]
        y_train = df_tra[dataset.continuous_label_name]
        y_test_c = df_tst_c[dataset.continuous_label_name]
        y_test_b = df_tst_b[dataset.binary_label_name]
        y_test_norm = normalisation(y_test_c, min_v, max_v)
        y_test_ordinal_normalized = pd.DataFrame(y_test_norm, columns=[dataset.continuous_label_name])


        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test_b.reset_index(drop=True, inplace=True)
        y_test_c.reset_index(drop=True, inplace=True)
        y_test_ordinal_normalized.reset_index(drop=True, inplace=True)

        A_train = X_train[a_name].astype('int')
        A_test = X_test[a_name].astype('int')

        column_predicted = dataset.outcome_label + '_predicted'

        y_test = df_tst_c[dataset.continuous_label_name]
        y_test.reset_index(drop=True, inplace=True)
        # favorable_label = dataset.favorable_label_binary

        if not mitigation:
            clf = lr_estimator.fit(X_train, y_train)
            results_ = clf.predict(X_test)
            results = pd.DataFrame(results_, columns=[column_predicted])
            # ordinal_output = dataset.compute_continuous_output(pd.DataFrame(results_))
            mae.append(mean_absolute_error(y_test_c, results[column_predicted]))

        else:
            # moment = SquareLoss(min(y_ordinal[dataset.continuous_label_name]),
            #                     max(y_ordinal[dataset.continuous_label_name]))
            bgl = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.01)

            grid_search_red = GridSearch(estimator=lr_estimator,
                                         constraints=bgl,
                                         grid_size=70)
            y_test_ = y_test_c
            grid_search_red.fit(X_train, y_train, sensitive_features=A_train)
            predictors = grid_search_red.predictors_
            gs_pred = grid_search_red.predict(X_test)
            results = pd.DataFrame(gs_pred, columns=[column_predicted])
            mae.append(mean_absolute_error(y_test_c, results[column_predicted]))




        metrics_.append(compute_metrics(y_test, results[column_predicted]))
        sp_mi_.append(sp_mi(X_test, y_test, results[column_predicted], index_mi))


        #sp_ks_.append(compute_sp_ks(results[column_predicted], X_test[dataset.protected_att_name[0]].values.tolist(),
        #                            dataset.privileged_classes[0][0]))
        sp_avg_outcome_.append(
            compute_sp_avg_outcome(results[column_predicted], X_test[dataset.protected_att_name[0]].values.tolist(),
                                   dataset.privileged_classes[0][0]))
        #ea_.append(compute_ea(y_test, results[column_predicted],  # [dataset.continuous_label_name]
        #                      X_test[dataset.protected_att_name[0]].values.tolist(), dataset.privileged_classes[0][0]))
        #didi_.append(didi(y_test, X_test[dataset.protected_att_name[0]].values.tolist(), # [dataset.continuous_label_name]
        #                  dataset.privileged_classes[0][0], min_v, max_v))
        #didi_predicted_.append(didi(results[column_predicted], X_test[dataset.protected_att_name[0]].values.tolist(),
        #                            dataset.privileged_classes[0][0], min_v, max_v))
        print('normalized')


        results_normalized = normalisation(results[column_predicted], min_v, max_v)
        results_n = pd.DataFrame(results_normalized, columns=[column_predicted])

        metrics_n.append(compute_metrics(y_test_ordinal_normalized, results_normalized))
        sp_mi_n.append(sp_mi(X_test, y_test_ordinal_normalized, results_normalized, index_mi))
        #sp_ks_n.append(compute_sp_ks(results_normalized, X_test[dataset.protected_att_name[0]].values.tolist(),
        #                             dataset.privileged_classes[0][0]))
        sp_avg_outcome_n.append(
            compute_sp_avg_outcome(results_normalized, X_test[dataset.protected_att_name[0]].values.tolist(),
                                   dataset.privileged_classes[0][0]))
        #ea_n.append(compute_ea(y_test_ordinal_normalized[dataset.continuous_label_name], results_normalized,
        #                       X_test[dataset.protected_att_name[0]].values.tolist(), dataset.privileged_classes[0][0]))
        mae_n.append(mean_absolute_error(y_test_ordinal_normalized[dataset.continuous_label_name], results_normalized))

        #didi_n.append(didi(y_test_ordinal_normalized[dataset.continuous_label_name],
        #                   X_test[dataset.protected_att_name[0]].values.tolist(), dataset.privileged_classes[0][0],
        #                   min_v, max_v))
        #didi_predicted_n.append(didi(results_normalized, X_test[dataset.protected_att_name[0]].values.tolist(),
        #                             dataset.privileged_classes[0][0], min_v, max_v))
        #deo.append(compute_deo(y_test_c, results_n[column_predicted], # [dataset.continuous_label_name]
        #                       X_test[dataset.protected_att_name[0]].values))

        # transforming the output to binary values
        binary_c = dataset.continuous_to_binary(y_test_c)  # [dataset.continuous_label_name]
        y_test_c = pd.DataFrame(binary_c, columns=[dataset.continuous_label_name])
        y_test_c.reset_index(drop=True, inplace=True)
        results[column_predicted] = dataset.continuous_to_binary(results[column_predicted])


        results_cm = cm(y_test_c, results)
        print(results_cm)
        acc = accuracy_score(y_test_c, results)
        print(acc)
        accuracies.append(acc)

        # NOTE check aunque el df sea ordinal o continuo hay que calcular las métricas con binario
        ds_tra = get_aif_dataset(X_test, y_test_b, label=dataset.binary_label_name,
                                 protected_attribute_names=dataset.protected_att_name,
                                 privileged_classes=dataset.privileged_classes,
                                 favorable_classes=dataset.favorable_label_binary)

        res, classm, predm = fair_metrics(ds_tra, results[column_predicted], dataset.privileged_groups,
                                          dataset.unprivileged_groups)
        statistical_parity_difference.append(predm.statistical_parity_difference())
        disparate_impact.append(predm.disparate_impact())
        equal_opportunity_difference.append(classm.equal_opportunity_difference())
        average_odds_difference.append(classm.average_odds_difference())
        average_predictive_value_difference.append(funct_average_predictive_value_difference(classm))
        false_discovery_rate_difference.append(classm.false_discovery_rate_difference())
        print(res)

    mae_prob = mae
    diff_err_prob = diff_err

    dict_metrics = {'accuracies': accuracies,
                    'statistical_parity_difference': statistical_parity_difference,
                    'disparate_impact': disparate_impact,
                    'equal_opportunity_difference': equal_opportunity_difference,
                    'average_odds_difference': average_odds_difference,
                    'average_predictive_value_difference': average_predictive_value_difference,
                    'false_discovery_rate_difference': false_discovery_rate_difference,
                    'mean_absolute_error': mae,
                    'mean_absolute_error_prob': mae_prob,
                    'metrics': metrics_,
                    'sp_mi': sp_mi_,
                    # 'sp_ks': sp_ks_,
                    'sp_avg_outcome': sp_avg_outcome_,
                    #'ea': ea_,
                    #'didi': didi_,
                    #'didi_predicted': didi_predicted_,
                    'metrics_n': metrics_n,
                    'sp_mi_n': sp_mi_n,
                    # 'sp_ks_n': sp_ks_n,
                    'sp_avg_outcome_n': sp_avg_outcome_n,
                    #'ea_n': ea_n,
                    #'didi_n': didi_n,
                    #'didi_predicted_n': didi_predicted_n,
                    #'deo': deo,
                    'mae_n': mae_n
                    }
    print(dict_metrics)
    df_metrics = pd.DataFrame(dict_metrics)
    return df_metrics
