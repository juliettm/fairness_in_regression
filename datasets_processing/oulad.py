# preprocesing
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from datasets_processing._dataset import BaseDataset, get_the_middle


class OuladDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'oulad'
        self._att = att
        self.preprocess()

    def preprocess(self):

        df_base = pd.read_csv("datasets/oulad/preprocessed_oulad.csv", sep=',', index_col=0)

        # this is always binary
        df_base['binary_Grade'] = df_base['Grade']

        # processed_df.loc[raw_df['gender'] == 'M', 'Sex'] = 1
        df_base.loc[df_base['Sex'] == 2, 'Sex'] = 0

        target_variable_ordinal = 'Grade'
        target_variable_binary = 'binary_Grade'


        self._ds = df_base

        self._explanatory_variables = ['Sex', 'Disability', 'Region', 'CodeModule', 'CodePresentation',
                                        'HighestEducation', 'IMDBand', 'AgeGroup', 'NumPrevAttempts',
                                        'StudiedCredits']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal  # [1-9]

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        # TODO change this to correspond with the paper values
        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # male
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

