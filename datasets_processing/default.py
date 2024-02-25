# preprocesing
import pandas as pd

from datasets_processing._dataset import BaseDataset, get_the_middle
from sklearn import preprocessing


class DefaultDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'default'
        self._att = att
        self.preprocess()

    def preprocess(self):

        df_base = pd.read_excel("datasets/default/default.xls",  index_col=0, header=0)

        # this is always binary
        # (Yes = 1, No = 0)
        df_base.rename(columns={'default_payment_next_month': 'default'}, inplace=True)
        df_base['binary_default'] = df_base['default']

        # protected att Gender (1 = male; 2 = female)
        df_base.loc[df_base['SEX'] == 2, 'SEX'] = 0

        columns_with_negatives = [col for col in df_base.columns if df_base[col].lt(0).any()]
        rest = df_base.columns.difference(columns_with_negatives)

        x = df_base[columns_with_negatives].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_pos = pd.DataFrame(x_scaled, columns=columns_with_negatives)

        df_rest = df_base[rest]
        df_rest.reset_index(drop=True, inplace=True)

        df_def = pd.concat([df_pos, df_rest], axis=1)

        target_variable_ordinal = 'default'
        target_variable_binary = 'binary_default'

        self._ds = df_def

        self._explanatory_variables = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
                                        'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                                        'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                                        'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # Male
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
