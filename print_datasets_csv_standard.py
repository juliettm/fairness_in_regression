import os
import sys
import warnings

import numpy as np
import pandas as pd

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
from datasets_processing.academic import AcademicDataset
from datasets_processing.adult import AdultDataset
from datasets_processing.arrhytmia import ArrhythmiaDataset
from datasets_processing.bank import BankDataset
from datasets_processing.catalunya import CatalunyaDataset
from datasets_processing.credit import CreditDataset
from datasets_processing.default import DefaultDataset
from datasets_processing.diabetes_w import DiabetesWDataset
from datasets_processing.diabetes import DiabetesDataset
from datasets_processing.dutch import DutchDataset
from datasets_processing.german import GermanDataset
from datasets_processing.heart import HeartDataset
from datasets_processing.hrs import HrsDataset
from datasets_processing.kdd_census import KddCensusDataset
from datasets_processing.nursery import NurseryDataset
from datasets_processing.oulad import OuladDataset
from datasets_processing.ricci import RicciDataset
from datasets_processing.synthetic_athlete import SyntheticAthleteDataset
from datasets_processing.synthetic_disease import SyntheticDiseaseDataset
from datasets_processing.toy import ToyDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from methods.linear_regression import apply_lin_regression
from methods.logistic_regression import apply_log_regression
from methods.ordinal_regression import apply_ord_regression

warnings.filterwarnings("ignore")
# pd.set_option('max_columns', None)

to_insert = os.getcwd()
# to import utils
sys.path.append(to_insert)

def get_matrices(df_name, seed, output, binary_label_name, continuous_label_name, protected_att_name):
    """
    Split dataframe into train and test.
    """

    df = pd.read_csv('data/' + df_name + '.csv', sep=',')

    y = df.iloc[:, -2:]

    scaler = StandardScaler()
    data_in = df.iloc[:, :-2]
    X = pd.DataFrame(scaler.fit_transform(data_in), columns=df.iloc[:, :-2].columns)
    X[protected_att_name] = data_in[protected_att_name]


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = seed)
    return X_train, X_val, X_test, y_train[binary_label_name], y_val[binary_label_name], y_test[binary_label_name], y_train[continuous_label_name], y_val[continuous_label_name], y_test[continuous_label_name]

def write_train_val_test(df_name, seed, X_train, X_val, X_test, y_tr_b, y_v_b, y_tst_b, y_tr_c, y_v_c, y_tst_c, output):
    # Specify the path where you want to create the folder
    folder_path = './data/train_val_test_standard/' + df_name

    # Check if the folder exists, and if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")

    train_b = X_train.copy()
    train_b['y'] = y_tr_b.tolist()
    train_b.to_csv('./data/train_val_test_standard/' + df_name + '/' + df_name + '_output_' + output + '_train_seed_' + str(seed) + '.csv', index = False)
    train_c = X_train.copy()
    train_c['y'] = y_tr_c.tolist()
    train_c.to_csv('./data/train_val_test_standard/' + df_name + '/' + df_name + '_output_continuous_train_seed_' + str(seed) + '.csv', index=False)

    val_b = X_val.copy()
    val_b['y'] = y_v_b.tolist()
    val_b.to_csv('./data/train_val_test_standard/' + df_name + '/' + df_name + '_output_' + output + '_val_seed_' + str(seed) + '.csv', index = False)
    val_c = X_val.copy()
    val_c['y'] = y_v_c.tolist()
    val_c.to_csv('./data/train_val_test_standard/' + df_name + '/' + df_name + '_output_continuous_val_seed_' + str(seed) + '.csv', index=False)

    test_b = X_test.copy()
    test_b['y'] = y_tst_b.tolist()
    test_b.to_csv('./data/train_val_test_standard/' + df_name + '/' + df_name + '_output_' + output + '_test_seed_' + str(seed) + '.csv', index = False)
    test_c = X_test.copy()
    test_c['y'] = y_tst_c.tolist()
    test_c.to_csv('./data/train_val_test_standard/' + df_name + '/' + df_name + '_output_continuous_test_seed_' + str(seed) + '.csv', index = False)


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
                LsatDataset('race', outcome_type=outcome),
                #LsatDataset('gender', outcome_type=outcome),
                StudentDataset('sex', outcome_type=outcome)
                ]
                # # New
                # AcademicDataset('ge', outcome_type=outcome),
                # AdultDataset('Sex', outcome_type=outcome),
                # ArrhythmiaDataset('sex', outcome_type=outcome),
                # BankDataset('AgeGroup', outcome_type=outcome),
                # CatalunyaDataset('foreigner', outcome_type=outcome),
                # CreditDataset('sex', outcome_type=outcome),
                # DefaultDataset('SEX', outcome_type=outcome),
                # DiabetesDataset('Sex', outcome_type=outcome),
                # DiabetesWDataset('Age', outcome_type=outcome),
                # DutchDataset('Sex', outcome_type=outcome),
                # GermanDataset('Sex', outcome_type=outcome),
                # HeartDataset('Sex', outcome_type=outcome),
                # HrsDataset('gender', outcome_type=outcome),
                # KddCensusDataset('Sex', outcome_type=outcome),
                # NurseryDataset('finance', outcome_type=outcome),
                # OuladDataset('Sex', outcome_type=outcome),
                # RicciDataset('Race', outcome_type=outcome),
                # SyntheticAthleteDataset('Sex', outcome_type=outcome),
                # SyntheticDiseaseDataset('Age', outcome_type=outcome),
                # ToyDataset('sst', outcome_type=outcome)
                # ]
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
                LsatDataset('race', outcome_type=outcome),
                StudentDataset('sex', outcome_type=outcome)
                ]
    else:
        raise AssertionError('not a valid outcome: ', outcome)

outcomes = ['binary', 'ordinal', 'continuous']


for outcome in outcomes:
    datasets = get_datasets(outcome)
    #datasets = [LsatDataset('race', outcome_type=outcome)]
    for dataset in datasets:
        print("Executing...", dataset, outcome)
        ds_columns = dataset._explanatory_variables + [dataset._continuous_label_name] + [dataset._binary_label_name]

        dataset.ds[ds_columns].to_csv('./data/' + dataset._name + '.csv', index = False)

        set_seed_base = 100
        n_runs = 10

        for run in range(n_runs):
            set_seed = set_seed_base + run

            # write datasets
            X_tr, X_v, X_tst, y_tr_b, y_v_b, y_tst_b, y_tr_c, y_v_c, y_tst_c = get_matrices(dataset._name, set_seed, outcome, dataset._binary_label_name, dataset._continuous_label_name, dataset._protected_att_name[0])
            write_train_val_test(dataset._name, set_seed, X_tr, X_v, X_tst, y_tr_b, y_v_b, y_tst_b, y_tr_c, y_v_c, y_tst_c, outcome)