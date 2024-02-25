# preprocesing
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from datasets_processing._dataset import BaseDataset, get_the_middle


class CreditDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'credit'
        self._att = att
        self.preprocess()

    def preprocess(self):

        df_base = pd.read_csv("datasets/credit/preprocessed_credit.csv", sep=',', index_col=0)

        # this is always binary
        no_default = df_base['NoDefaultNextMonth (label)']
        df_base.rename(columns={"isMale": "sex"}, inplace=True)
        df_base.drop(columns=['NoDefaultNextMonth (label)'], inplace=True)
        df_base['NoDefault'] = no_default
        df_base['binary_NoDefault'] = df_base['NoDefault']

        target_variable_ordinal = 'NoDefault'
        target_variable_binary = 'binary_NoDefault'


        self._ds = df_base

        self._explanatory_variables = [ 'sex', 'isMarried', 'AgeGroup',
                                        'EducationLevel', 'MaxBillAmountOverLast6Months',
                                        'MaxPaymentAmountOverLast6Months',
                                        'MonthsWithZeroBalanceOverLast6Months',
                                        'MonthsWithLowSpendingOverLast6Months',
                                        'MonthsWithHighSpendingOverLast6Months', 'MostRecentBillAmount',
                                        'MostRecentPaymentAmount', 'TotalOverdueCounts', 'TotalMonthsOverdue',
                                        'HasHistoryOfOverduePayments']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        # TODO change this to correspond with the paper values
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
