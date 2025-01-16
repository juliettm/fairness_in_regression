# Import necessary libraries
import pandas as pd
import os

from sklearn.tree import DecisionTreeClassifier
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
from datasets_processing.wine import WineDataset

import matplotlib.pyplot as plt
import seaborn as sns

from datasets_processing.aif360datset import get_aif_dataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from sklearn.model_selection import train_test_split


to_insert = os.getcwd()

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
                #TicDataset('religion', outcome_type=outcome),
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
                #TicDataset('religion', outcome_type=outcome),
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


data_SP = {}
thres_SP = {}

for dataset in datasets:
    print(dataset._name)
    df = dataset._ds[dataset._explanatory_variables + [dataset._continuous_label_name]]
    output_var = dataset._continuous_label_name

    y_ord = dataset.ds[dataset.continuous_label_name]
    min_v = min(y_ord)
    max_v = max(y_ord)
    STEP = (max_v - min_v)/10

    # Create a copy of the original DataFrame for discretized output variables
    df_discretized_output = df.copy()

    thresholds = ['STEP1', 'STEP2', 'STEP3', 'STEP4', 'STEP5', 'STEP6', 'STEP7', 'STEP8', 'STEP9']
    thresholds_values = {}
    i = min_v
    for threshold in thresholds:
        threshold_value = i + STEP
        df_discretized_output[threshold] = (df[output_var] > threshold_value).astype(int)
        #print(threshold_value)
        thresholds_values[threshold] = threshold_value
        i += STEP
    #print(thresholds_values)
    thres_SP[dataset._name] = thresholds_values

    X = df_discretized_output[dataset._explanatory_variables]
    y = df_discretized_output[thresholds]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sp_values = {}

    for threshold in thresholds:
        # print("Computind SP for threshold", threshold)

        clf = DecisionTreeClassifier(random_state=42)

        model = clf.fit(X_train, y_train[threshold])
        predicted = model.predict(X_test)


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

    # print(sp_values)
    data_SP[dataset._name] = sp_values


datasets = get_datasets('ordinal')
# datasets = [CrimeDataset('race', outcome_type='continuous')]
# datasets = [DrugsDataset('Gender', outcome_type='ordinal')]

for dataset in datasets:
    print(dataset._name)
    df = dataset._ds[dataset._explanatory_variables + [dataset._continuous_label_name]]
    output_var = dataset._continuous_label_name

    y_ord = dataset.ds[dataset.continuous_label_name]
    min_v = min(y_ord)
    max_v = max(y_ord)
    STEP = (max_v - min_v)/10

    # Create a copy of the original DataFrame for discretized output variables
    df_discretized_output = df.copy()

    thresholds = ['STEP1', 'STEP2', 'STEP3', 'STEP4', 'STEP5', 'STEP6', 'STEP7', 'STEP8', 'STEP9']
    thresholds_values = {}
    i = min_v
    for threshold in thresholds:
        threshold_value = i + STEP
        df_discretized_output[threshold] = (df[output_var] > threshold_value).astype(int)
        #print(threshold_value)
        thresholds_values[threshold] = threshold_value
        i += STEP

    thres_SP[dataset._name] = thresholds_values

    X = df_discretized_output[dataset._explanatory_variables]
    y = df_discretized_output[thresholds]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sp_values = {}

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

    #print(sp_values)
    data_SP[dataset._name] = sp_values

print(data_SP)
# Convert the dictionary to a DataFrame
df = pd.DataFrame(data_SP).T
print(df)

# Convert the dictionary to a DataFrame
df_T = pd.DataFrame(thres_SP).T
print(df_T)

exit(0)

# Set a color palette (one color for each row)
palette = sns.color_palette("tab20", len(df))

# Create the coordinate plot
plt.figure(figsize=(10, 6))
for idx, (dataset, row) in enumerate(df.iterrows()):
    plt.plot(row.index, row.values, marker='o', linestyle='-', color=palette[idx], label=dataset)

# Add labels and legend
# plt.xlabel("Steps")
plt.ylabel("SP Classification")
plt.title("SP values across different thresholds for each dataset")
plt.xticks(rotation=45)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), ncol=1)
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.savefig('plots/SP_by_steps.png')
