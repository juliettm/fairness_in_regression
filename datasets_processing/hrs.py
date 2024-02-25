# preprocesing
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from datasets_processing._dataset import BaseDataset, get_the_middle

race_map = {'NHW': 0,
            'NHB': 1,
            'Hispanic': 2,
            'Other': 3}
race_map = {'NHW': 1,
            'NHB': 0,
            'Hispanic': 0,
            'Other': 0}


class HrsDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'hrs'
        self._att = att
        self.preprocess()

    def preprocess(self):

        df = pd.read_csv("datasets/hrs/HRS_ADL_IADL.csv", sep=',', index_col=0)
        df = df.drop(['year', 'BIRTHYR', 'HISPANIC', 'race'], axis=1)
        df.dropna(axis=0, inplace=True, how='any')
        df.reset_index(drop=True, inplace=True)

        cat = pd.DataFrame(index=df.index)
        cat['marriage'] = (df['marriage'].values == 'Not Married').astype(float)
        cat['gender'] = (df['gender'] == 'Female').astype(float)
        cat['race'] = np.array([race_map[r] for r in df['race.ethnicity']])

        num = df.drop(['marriage', 'gender', 'score', 'race.ethnicity'], axis=1)
        target = (df['score'] < 2).astype(float)

        # all together
        df_base = num.join(cat).join(target)

        # this is always binary
        df_base['binary_score'] = target

        # print(df_base.groupby('score')['score'].value_counts())
        # print(df_base.groupby('race')['race'].value_counts())
        # print(df_base.columns)

        target_variable_ordinal = 'score'
        target_variable_binary = 'binary_score'

        self._ds = df_base

        self._explanatory_variables = ['AGE', 'educa', 'networth', 'cognition_catnew', 'bmi', 'hlthrte',
                                        'bloodp', 'diabetes', 'cancer', 'lung', 'heart', 'stroke', 'pchiat',
                                        'arthrit', 'fall', 'pain', 'A1c_adj', 'CRP_adj', 'CYSC_adj', 'HDL_adj',
                                        'TC_adj', 'marriage', 'gender', 'race']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal  # [1-9]

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # caucasian
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
