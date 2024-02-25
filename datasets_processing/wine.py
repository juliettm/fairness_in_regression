import pandas as pd

from datasets_processing._dataset import BaseDataset, get_the_middle


class WineDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'ordinal'
        self._name = 'wine'
        self._att = att
        self.preprocess()

    def preprocess(self):
        df_red = pd.read_csv("datasets/wine/winequality-red.csv", sep=';')
        df_white = pd.read_csv("datasets/wine/winequality-white.csv", sep=';')

        # red -> 0
        # white -> 1
        df_red['color'] = 0
        df_white['color'] = 1

        dataset = pd.concat([df_red, df_white], ignore_index=True)

        # We left out category 9 wine - there only white wines there
        dataset = dataset.drop(dataset[dataset.quality == 9].index)
        dataset.reset_index(inplace=True, drop=True)

        # computing the binary outcome
        dataset['quality_binary'] = dataset['quality'] > 5
        dataset['quality_binary'] = dataset['quality_binary'].astype(int)

        dataset.reset_index(drop=True, inplace=True)

        self._ds = dataset

        self._explanatory_variables = ['fixedacidity', 'volatileacidity', 'citricacid', 'residualsugar',
                                       'chlorides', 'freesulfurdioxide', 'totalsulfurdioxide', 'density',
                                       'pH', 'sulphates', 'alcohol', 'color']

        if self.outcome_type == 'binary':
            self._outcome_label = 'quality_binary'  # [0, 1]
        else:
            self._outcome_label = 'quality'  # [3-8]

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [6, 7, 8]

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # white
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = 'quality'
        self._binary_label_name = 'quality_binary'

        self._cut_point = 6  # is >= all above is 1
        self._non_favorable_label_continuous = [3, 4, 5]

        self._fav_dict = self.assign_ranges_to_ordinal(fav=True)
        self._nonf_dict = self.assign_ranges_to_ordinal(fav=False, sort_desc=True)

        self._middle_fav = get_the_middle(self.favorable_label_continuous)
        self._middle_non_fav = get_the_middle(self.non_favorable_label_continuous)
