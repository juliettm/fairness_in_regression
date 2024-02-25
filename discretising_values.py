# Import necessary libraries
import pandas as pd
import numpy as np
import os
import sys
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from datasets_processing.compas import CompasDataset
from datasets_processing.crime import CrimeDataset
from datasets_processing.drugs import DrugsDataset
from datasets_processing.insurance import InsuranceDataset
from datasets_processing.lsat import LsatDataset
from datasets_processing.obesity import ObesityDataset
from datasets_processing.older_adults import OlderAdultsDataset
from datasets_processing.parkinson import ParkinsonDataset
from datasets_processing.singles import SinglesDataset
from datasets_processing.student import StudentDataset
from datasets_processing.tic import TicDataset
from datasets_processing.wine import WineDataset

import matplotlib.pyplot as plt
import seaborn as sns

from datasets_processing.aif360datset import get_aif_dataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from sklearn.model_selection import train_test_split


to_insert = os.getcwd()
# to import utils
# sys.path.append('/Users/juls/Desktop/BIAS- regression/compas_analysis/')

# compute fairness metrics for two classes
def fair_metrics(dataset, y_predicted, privileged_groups, unprivileged_groups):
    dataset_pred = dataset.copy()
    dataset_pred.labels = y_predicted

    classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)

    return round(metric_pred.statistical_parity_difference(), 3)

def get_datasets(outcome):
    if outcome == 'binary':
        return [CompasDataset('race', outcome_type=outcome),
                WineDataset('color', outcome_type=outcome),
                SinglesDataset('sex', outcome_type=outcome),
                TicDataset('religion', outcome_type=outcome),
                ObesityDataset('Gender', outcome_type=outcome),
                DrugsDataset('Gender', outcome_type=outcome),
                # Continuous
                InsuranceDataset('sex', outcome_type=outcome),
                ParkinsonDataset('sex', outcome_type=outcome),
                CrimeDataset('race', outcome_type=outcome),
                OlderAdultsDataset('sex', outcome_type=outcome),
                #LsatDataset('race', outcome_type=outcome),
                LsatDataset('gender', outcome_type=outcome),
                StudentDataset('sex', outcome_type=outcome)
                ]
    elif outcome == 'ordinal':
        return [CompasDataset('race', outcome_type=outcome),
                WineDataset('color', outcome_type=outcome),
                SinglesDataset('sex', outcome_type=outcome),
                TicDataset('religion', outcome_type=outcome),
                ObesityDataset('Gender', outcome_type=outcome),
                DrugsDataset('Gender', outcome_type=outcome)
                ]
    elif outcome == 'continuous':
        return [InsuranceDataset('sex', outcome_type=outcome),
                ParkinsonDataset('sex', outcome_type=outcome),
                CrimeDataset('race', outcome_type=outcome),
                OlderAdultsDataset('sex', outcome_type=outcome),
                #LsatDataset('race', outcome_type=outcome),
                LsatDataset('gender', outcome_type=outcome),
                StudentDataset('sex', outcome_type=outcome)
                ]
    else:
        raise AssertionError('not a valid outcome: ', outcome)


datasets = get_datasets('continuous')
# datasets = [CrimeDataset('race', outcome_type='continuous')]
# datasets = [DrugsDataset('Gender', outcome_type='ordinal')]
figure = False


for dataset in datasets:
    print(dataset._name)
    df = dataset._ds[dataset._explanatory_variables + [dataset._continuous_label_name]]
    output_var = dataset._continuous_label_name

    # Create a copy of the original DataFrame for discretized output variables
    df_discretized_output = df.copy()

    # 1. Thresholding with Mean
    mean_threshold = df[output_var].mean()
    df_discretized_output['output_var_binary_mean'] = (df[output_var] > mean_threshold).astype(int)

    # 1. Thresholding with Median
    median_threshold = df[output_var].median()
    df_discretized_output['output_var_binary_median'] = (df[output_var] > median_threshold).astype(int)

    # 2. Equal Frequency (Quantile) Binning
    # df_discretized_output['output_var_eq_freq'] = pd.qcut(df[output_var], q=2, labels=[0, 1])

    # 3. K-Means Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    df_discretized_output['output_var_kmeans'] = kmeans.fit_predict(df[[output_var]])

    # 4. Decision Tree Classifier
    tree = DecisionTreeRegressor(max_depth=1, random_state=42)  # Max depth 1 for binary output
    tree.fit(df.iloc[:, :-1], df[output_var])  # Exclude the original output variable
    df_discretized_output['output_var_tree'] = tree.apply(df.iloc[:, :-1])

    # 5. Expert
    df_discretized_output['expert'] = dataset._ds[dataset._binary_label_name]

    thresholds = ['output_var_binary_mean', 'output_var_binary_median', 'output_var_kmeans', 'expert']
    thresholds_values = {'output_var_binary_mean': mean_threshold, 'output_var_binary_median': median_threshold,  'expert': dataset._cut_point}
    sp_values = {}
    print(thresholds_values)

    X = df_discretized_output[dataset._explanatory_variables]
    y = df_discretized_output[thresholds]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for threshold in thresholds:
        # print("Computind SP for threshold", threshold)

        clf = DecisionTreeClassifier(random_state=42)

        model = clf.fit(X_train, y_train[threshold])
        predicted = model.predict(X_test)
        #df_discretized_output[threshold + '_predicted'] = predicted


        ds_ = get_aif_dataset(X_test,
                              y_test[threshold],
                              label=threshold,
                              protected_attribute_names=dataset._protected_att_name,
                              privileged_classes=dataset._privileged_classes,
                              favorable_classes=dataset._favorable_label_binary)

        sp = fair_metrics(ds_,
                          predicted,
                          dataset.privileged_groups,
                          dataset.unprivileged_groups)

        sp_values[threshold] = sp

    print(sp_values)

    # Plot the distribution of the continuous output by protected attribute
    if figure:
        # Create a DataFrame
        data = pd.DataFrame({'Output': dataset._ds[dataset._continuous_label_name], 'Protected': dataset._ds[dataset._protected_att_name[0]]})

        # Plot the data
        plt.figure(figsize=(8, 6))
        if dataset.outcome_type == 'continuous':
            ax = sns.histplot(data=data, x='Output', hue='Protected', kde=True, common_norm=False)
        else:
            ax = sns.countplot(data=data,  x='Output', hue='Protected')
        # plt.xlabel('Coke Recency (ordinal intervals)', fontsize=16)
        plt.xlabel('Crimes (%)', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.title('Distribution of Continuous Output by Protected Attribute', fontsize=16)
        # plt.title('Histogram of the Output by Protected Attribute', fontsize=16)
        # plt.legend(title='Protected Attribute', labels=['Group 0', 'Group 1'])
        # plt.legend(title='Gender', labels=['Male', 'Female'], fontsize=14)
        plt.legend(title='Race', fontsize=14, labels=['White', 'Other'])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Add reference lines for each value in the dictionary
        for key, value in thresholds_values.items():
            ax.axvline(x=round(value, 2), color='black', linestyle='--', label=f'{key}: {round(value, 2)}')

        # Adjust the layout to minimize borders
        plt.tight_layout()

        plt.savefig('plots/discretising/{}_distribution.pdf'.format(dataset._name))

