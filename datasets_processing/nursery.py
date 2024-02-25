# preprocesing
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from datasets_processing._dataset import BaseDataset, get_the_middle


column_mapping = {
'class': {'not_recom':0, 'recommend':1, 'very_recom':1, 'priority':1, 'spec_prior':1},
'parents': {'usual':1, 'pretentious':2, 'great_pret':3},
'has_nurs': {'proper':1, 'less_proper':2, 'improper':3, 'critical':4, 'very_crit':5},
'form':   {'complete':1, 'completed':2, 'incomplete':3, 'foster':4},
'children': {'1':1, '2':2, '3':3, 'more':4},
'housing': {'convenient':3, 'less_conv':2, 'critical':1},
'finance': {'convenient':1, 'inconv':0},
'social':  {'nonprob':1, 'slightly_prob':2, 'problematic':3},
'health': {'recommended':1, 'priority':2, 'not_recom':3}
}

columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']

class NurseryDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'nursery'
        self._att = att
        self.preprocess()

    def preprocess(self):

        data = pd.read_csv("datasets/nursery/nursery.data", names=columns)

        df_base = pd.DataFrame()
        # Convert columns to numeric, if possible
        for col, mapping in column_mapping.items():
            df_base[col] = data[col].map(mapping)



        target_variable_ordinal = 'class'
        target_variable_binary = 'binary_class'

        clas_output = df_base['class']
        df_base.drop(['class'], axis=1, inplace=True)
        df_base['class'] = clas_output

        # this is always binary
        df_base['binary_class'] = df_base['class']

        self._ds = df_base

        self._explanatory_variables = ['parents', 'has_nurs', 'form', 'children', 'housing',
                                        'finance', 'social', 'health']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        # TODO change this to correspond with the paper values
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
