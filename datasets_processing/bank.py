# preprocesing
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from datasets_processing._dataset import BaseDataset, get_the_middle


class BankDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'bank'
        self._att = att
        self.preprocess()

    def preprocess(self):
        # TODO aqui también puedo eliminar age = 1

        df_base = pd.read_csv("datasets/bank/preprocessed_bank.csv", sep=',', index_col=0)

        # this is always binary
        df_base['binary_Subscribed'] = df_base['Subscribed']
        df_base['AgeGroup'] = (df_base['AgeGroup'] != 3).astype(float)

        df_base.dropna(axis=0, how='any', inplace=True)
        df_base.reset_index(drop=True, inplace=True)

        target_variable_ordinal = 'Subscribed'
        target_variable_binary = 'binary_Subscribed'

        self._ds = df_base

        self._explanatory_variables = [ 'Default', 'Housing', 'Loan', 'AgeGroup', 'Balance', 'Day', 'Duration',
                                        'Campaign', 'Pdays', 'Previous', 'Job', 'MaritalStatus', 'Education',
                                        'Contact', 'Month', 'Poutcome']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        # TODO change this to correspond with the paper values
        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # <= 60
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = target_variable_ordinal
        self._binary_label_name = target_variable_binary

        self._cut_point = 1
        self._non_favorable_label_continuous = [0]

        self._fav_dict = self.assign_ranges_to_ordinal(fav=True)
        self._nonf_dict = self.assign_ranges_to_ordinal(fav=False, sort_desc=True)
        self._middle_fav = get_the_middle(self.favorable_label_continuous)
        self._middle_non_fav = get_the_middle(self.non_favorable_label_continuous)
