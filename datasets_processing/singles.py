import pandas as pd

from datasets_processing._dataset import BaseDataset, get_the_middle


class SinglesDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'ordinal'
        self._name = 'singles'
        self._att = att
        self.preprocess()

    def preprocess(self):
        columns = ['income', 'sex', 'marital', 'age', 'educ', 'occup', 'resid', 'dualInc', 'perInHou',
                   'under18', 'homeStatus', 'homeType', 'ethnic', 'language']
        df_s = pd.read_csv("datasets/singlesincome/data.csv", sep=',', names=columns)

        # Keep only the singles
        dataset = df_s.loc[df_s.marital == 5]
        dataset.drop(['marital'], axis=1, inplace=True)

        # Change codification for sex:
        gender = (df_s['sex'] == 1).astype(float)  # Male = 1
        dataset.drop(['sex'], axis=1, inplace=True)
        dataset['sex'] = gender

        # Remove nan values
        dataset.dropna(inplace=True)

        # annual income
        # 1.Less than $10, 000
        # 2. $10, 000to $14, 999
        # 3. $15, 000to $19, 999
        # 4. $20, 000to $24, 999
        # 5. $25, 000to $29, 999
        # 6. $30, 000to $39, 999
        # 7. $40, 000 to $49, 999
        # 8. $50, 000 to $74, 999
        # 9. $75, 000 or more
        # computing the binary outcome
        dataset['income_binary'] = dataset['income'] > 4
        dataset['income_binary'] = dataset['income_binary'].astype(int)

        dataset.reset_index(drop=True, inplace=True)

        self._ds = dataset

        self._explanatory_variables = ['sex', 'age', 'educ', 'occup', 'resid', 'dualInc', 'perInHou',
                                       'under18', 'homeStatus', 'homeType', 'ethnic', 'language']

        if self.outcome_type == 'binary':
            self._outcome_label = 'income_binary'  # [0, 1]
        else:
            self._outcome_label = 'income'  # [3-9]

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [5, 6, 7, 8, 9]

        # 1. Male
        # 2. Female
        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # male
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = 'income'
        self._binary_label_name = 'income_binary'

        self._cut_point = 5  # is >= all above is 1
        self._non_favorable_label_continuous = [1, 2, 3, 4]

        self._fav_dict = self.assign_ranges_to_ordinal(fav=True)
        self._nonf_dict = self.assign_ranges_to_ordinal(fav=False, sort_desc=True)

        self._middle_fav = get_the_middle(self.favorable_label_continuous)
        self._middle_non_fav = get_the_middle(self.non_favorable_label_continuous)
