# Ordinal Classification Implementation Source: Implementation of [Ordinal Classification Paper](
# https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf) using [Logistic Regression and SVM](
# https://github.com/sarvothaman/ordinal-classification)

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


from datasets_processing.aif360datset import get_aif_dataset
from methods.regression_measures import sp_mi, compute_sp_ks, compute_sp_avg_outcome, compute_ea, didi, compute_metrics, \
    normalisation, compute_deo

warnings.filterwarnings("ignore")


def get_model(protected_att, estimator):
    est = estimator
    return GridSearchReduction(prot_attr=protected_att,
                               estimator=est,
                               constraints=DemographicParity(),
                               loss="Absolute",
                               min_val=0,
                               max_val=1,
                               grid_size=10,
                               drop_prot_attr=False)


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


def apply_ord_regression(df_base, splits=10, mitigation=False):
    if df_base.outcome_type != 'ordinal':
        raise AssertionError('Only ordinal datasets allowed')

    lr_estimator = LogisticRegression(penalty='none', solver='lbfgs')
    # k_folds = StratifiedKFold(n_splits=splits, random_state=42, shuffle=True)

    # scaler = StandardScaler()
    # data_in = df_base.ds[df_base.explanatory_variables]
    # data_in = pd.DataFrame(scaler.fit_transform(data_in), columns=df_base.explanatory_variables)

    y_ord = df_base.ds[df_base.continuous_label_name]
    min_v = min(y_ord)
    max_v = max(y_ord)
    # y_norm = normalisation(y, min_v, max_v)
    # y_normalized = pd.DataFrame(y_norm, columns=[df_base.continuous_label_name])

    # X.reset_index(drop=True, inplace=True)
    # y.reset_index(drop=True, inplace=True)

    accuracies = []
    statistical_parity_difference = []
    disparate_impact = []
    equal_opportunity_difference = []
    average_odds_difference = []
    average_predictive_value_difference = []
    false_discovery_rate_difference = []
    sub_models = []
    mae = []
    mse = []
    mae_n = []
    diff_err = []
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
            "/Users/jsuarez/Documents/Personal/fairness_in_regression/data/train_val_test_standard/{}/{}_output_continuous_train_seed_{}.csv".format(
                df_base._name, df_base._name, seed))
        df_tst = pd.read_csv(
            "/Users/jsuarez/Documents/Personal/fairness_in_regression/data/train_val_test_standard/{}/{}_output_continuous_test_seed_{}.csv".format(
                df_base._name, df_base._name, seed))


        df_tra.rename(columns={'y': df_base.continuous_label_name}, inplace=True)
        df_tst.rename(columns={'y': df_base.continuous_label_name}, inplace=True)

        labels = df_base.ds[df_base.continuous_label_name].unique()
        labels.sort()
        labels = ["%02d" % n for n in labels]
        labels = labels[: -1]

        # Create "dummy" fields, one for each quality we are considering leaving out the last score (10)
        # Set this dummy field to 1, if the current score is greater than the score which the field represents
        new_variables = []
        for l in labels:
            var_name = df_base.continuous_label_name + '_' + str(l)
            # print(var_name)
            new_variables.append(var_name)
            df_tra[var_name] = 0
            df_tst[var_name] = 0
            df_tra.loc[(df_tra[df_base.continuous_label_name] > int(l)), var_name] = 1
            df_tst.loc[(df_tst[df_base.continuous_label_name] > int(l)), var_name] = 1

        model = lr_estimator

        X_train = df_tra[df_base.explanatory_variables+new_variables]
        X_test = df_tst[df_base.explanatory_variables+new_variables]
        y_train = df_tra[df_base.continuous_label_name]
        y_test = df_tst[df_base.continuous_label_name]
        y_test_norm = normalisation(y_test, min_v, max_v)
        y_test_ordinal_normalized = pd.DataFrame(y_test_norm, columns=[df_base.continuous_label_name])


        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        y_test_ordinal_normalized.reset_index(drop=True, inplace=True)
        X_test_ = X_test.copy()

        index_mi = df_base.explanatory_variables.index(df_base.protected_att_name[0])

        prob_array = []
        # Model for each label
        i = 0
        for var in new_variables:
            label = var[-2:]
            pred_label_old = pred_label_current if i > 0 else ''
            pred_label_current = 'predictions_' + label
            prob = 'pr' + label
            prob_array.append(prob)
            if not mitigation:
                clf_ = model.fit(X_train[X_train.columns.difference(new_variables)], X_train[var])
                X_test_[pred_label_current] = clf_.predict(X_test[X_test.columns.difference(new_variables)])

            else:
                ds_tra_ = get_aif_dataset(X_train[X_train.columns.difference(new_variables)], X_train[var],
                                          label=var,
                                          protected_attribute_names=df_base.protected_att_name,
                                          privileged_classes=df_base.privileged_classes,
                                          favorable_classes=df_base.favorable_label_binary
                                          )
                ds_tst_ = get_aif_dataset(X_test[X_test.columns.difference(new_variables)], X_test[var],
                                          label=var,
                                          protected_attribute_names=df_base.protected_att_name,
                                          privileged_classes=df_base.privileged_classes,
                                          favorable_classes=df_base.favorable_label_binary
                                          )
                clf_ = get_model(df_base.protected_att_name, lr_estimator).fit(ds_tra_)
                X_test_[pred_label_current] = clf_.predict(ds_tst_).labels

            if i == 0:
                X_test_[prob] = 1 - X_test_[pred_label_current]
            else:
                X_test_[prob] = X_test_[pred_label_old] - X_test_[pred_label_current]

            # TODO los comprueba siempre con el binario podemos tener métricas para múltiples clases
            ds_ = get_aif_dataset(X_test[X_test.columns.difference(new_variables)],
                                  X_train[var],
                                  label=var,
                                  protected_attribute_names=df_base.protected_att_name,
                                  privileged_classes=df_base.privileged_classes,
                                  favorable_classes=df_base.favorable_label_binary)

            res, classm, predm = fair_metrics(ds_, X_test_[pred_label_current], df_base.privileged_groups,
                                              df_base.unprivileged_groups)

            # res['rank'] = 1
            # TODO submodels tiene las métricas para cada ejecución intermedia.
            sub_models.append(res)
            i += 1

        # Model for the last label
        label = int(label) + 1
        label = "%02d" % label
        prob = 'pr' + label
        X_test_[prob] = X_test_[pred_label_current]
        prob_array.append(prob)

        X_test_["assigned_class_prob"] = X_test_[prob_array].idxmax(axis=1)
        predicted_variable_name = 'predicted_' + df_base.continuous_label_name
        X_test_[predicted_variable_name] = pd.to_numeric(X_test_["assigned_class_prob"].str[-2:])


        results_cm = cm(y_test, X_test_[predicted_variable_name])
        print(results_cm)
        acc = accuracy_score(y_test, X_test_[predicted_variable_name])
        print(acc)


        mae.append(mean_absolute_error(y_test, X_test_[predicted_variable_name]))
        diff_err.append(df_base.compute_diff_error(y_test, X_test_[predicted_variable_name]))

        metrics_.append(compute_metrics(y_test, X_test_[predicted_variable_name]))
        sp_mi_.append(sp_mi(X_test, y_test, X_test_[predicted_variable_name], index_mi))
        sp_ks_.append(
            compute_sp_ks(X_test_[predicted_variable_name], X_test[df_base.protected_att_name[0]].values.tolist(),
                          df_base.privileged_classes[0][0]))
        sp_avg_outcome_.append(compute_sp_avg_outcome(X_test_[predicted_variable_name],
                                                      X_test[df_base.protected_att_name[0]].values.tolist(),
                                                      df_base.privileged_classes[0][0]))
        ea_.append(
            compute_ea(y_test, X_test_[predicted_variable_name], X_test[df_base.protected_att_name[0]].values.tolist(),
                       df_base.privileged_classes[0][0]))
        didi_.append(
            didi(y_test, X_test[df_base.protected_att_name[0]].values.tolist(), df_base.privileged_classes[0][0], min_v,
                 max_v))
        didi_predicted_.append(
            didi(X_test_[predicted_variable_name], X_test[df_base.protected_att_name[0]].values.tolist(),
                 df_base.privileged_classes[0][0], min_v, max_v))

        results_normalized = normalisation(X_test_[predicted_variable_name], min_v, max_v)
        results_n = pd.DataFrame(results_normalized, columns=[predicted_variable_name])
        metrics_n.append(compute_metrics(y_test_ordinal_normalized, results_normalized))
        sp_mi_n.append(sp_mi(X_test, y_test_ordinal_normalized, results_normalized, index_mi))
        sp_ks_n.append(compute_sp_ks(results_normalized, X_test[df_base.protected_att_name[0]].values.tolist(),
                                     df_base.privileged_classes[0][0]))
        sp_avg_outcome_n.append(
            compute_sp_avg_outcome(results_normalized, X_test[df_base.protected_att_name[0]].values.tolist(),
                                   df_base.privileged_classes[0][0]))
        ea_n.append(compute_ea(y_test_ordinal_normalized[df_base.continuous_label_name], results_normalized,
                               X_test[df_base.protected_att_name[0]].values.tolist(), df_base.privileged_classes[0][0]))
        didi_n.append(didi(y_test_ordinal_normalized[df_base.continuous_label_name],
                           X_test[df_base.protected_att_name[0]].values.tolist(), df_base.privileged_classes[0][0],
                           min_v, max_v))
        didi_predicted_n.append(didi(results_normalized, X_test[df_base.protected_att_name[0]].values.tolist(),
                                     df_base.privileged_classes[0][0], min_v, max_v))
        mae_n.append(mean_absolute_error(y_test_ordinal_normalized, results_normalized))
        mse.append(mean_squared_error(y_test_ordinal_normalized, results_normalized))

        y_test = y_test.to_frame()
        deo.append(compute_deo(y_test[df_base.continuous_label_name], results_n[predicted_variable_name],
                               X_test[df_base.protected_att_name[0]].values))

        y_test_ = y_test
        y_pred = X_test_[predicted_variable_name].to_frame(name=predicted_variable_name)

        y_test_[df_base.continuous_label_name] = df_base.continuous_to_binary(y_test_[df_base.continuous_label_name])
        y_pred[predicted_variable_name] = df_base.continuous_to_binary(y_pred[predicted_variable_name])

        results_cm = cm(y_test_, y_pred)
        print(results_cm)
        acc = accuracy_score(y_test_, y_pred)
        print(acc)
        accuracies.append(acc)


        # NOTE aqui está convirtiendo antes de calcular por eso se queda con el continuous label name
        ds_ = get_aif_dataset(X_test, y_test, label=df_base.continuous_label_name,
                              # porque lo está convirtiendo en y_test[df_base.continuous_label_name]
                              protected_attribute_names=df_base.protected_att_name,
                              privileged_classes=df_base.privileged_classes,
                              favorable_classes=df_base.favorable_label_binary)

        res, classm, predm = fair_metrics(ds_, y_pred[predicted_variable_name], df_base.privileged_groups,
                                          df_base.unprivileged_groups)

        statistical_parity_difference.append(predm.statistical_parity_difference())
        disparate_impact.append(predm.disparate_impact())
        equal_opportunity_difference.append(classm.equal_opportunity_difference())
        average_odds_difference.append(classm.average_odds_difference())
        average_predictive_value_difference.append(funct_average_predictive_value_difference(classm))
        false_discovery_rate_difference.append(classm.false_discovery_rate_difference())

    dict_metrics = {'accuracies': accuracies,
                    'statistical_parity_difference': statistical_parity_difference,
                    'disparate_impact': disparate_impact,
                    'equal_opportunity_difference': equal_opportunity_difference,
                    'average_odds_difference': average_odds_difference,
                    'average_predictive_value_difference': average_predictive_value_difference,
                    'false_discovery_rate_difference': false_discovery_rate_difference,
                    'mean_absolute_error': mae,
                    'mae': mae,
                    'diff_err': diff_err,
                    'metrics': metrics_,
                    'sp_mi': sp_mi_,
                    'sp_ks': sp_ks_,
                    'sp_avg_outcome': sp_avg_outcome_,
                    'ea': ea_,
                    'didi': didi_,
                    'didi_predicted': didi_predicted_,
                    'metrics_n': metrics_n,
                    'sp_mi_n': sp_mi_n,
                    'sp_ks_n': sp_ks_n,
                    'sp_avg_outcome_n': sp_avg_outcome_n,
                    'ea_n': ea_n,
                    'didi_n': didi_n,
                    'didi_predicted_n': didi_predicted_n,
                    'deo': deo,
                    'mse_n': mse,
                    'mae_n': mae_n
                    }
    print(dict_metrics)
    ordinal_metrics = pd.DataFrame(dict_metrics)
    return ordinal_metrics
