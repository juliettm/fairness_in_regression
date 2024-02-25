import pandas as pd

# from aif360 import datasets as ds
from datasets_processing._dataset import BaseDataset


class OlderAdultsDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'countinuous'
        self._name = 'older-adults'
        self._att = att
        self.preprocess()

    def preprocess(self):
        df = pd.read_csv("datasets/olderadults/older_adults.csv", sep=';')

        columns = ['HEIGHT', 'WEIGHT', 'B-CURLS', '6MInWk', '6STEPS', '10METRE', 'TIMEDUP&GO', '5SittoStands',
                   'SITTOSTAND', 'GRIPLEFT', 'GRIPRIGHT', 'FR1', 'FR2', 'FR3', 'SEX', 'TOTALMISTAKES']

        data = df[columns]

        dataset = pd.DataFrame()
        for c in columns:
            if data[c].dtypes == 'O':
                new_col = data[c].apply(lambda x: float(x.replace(',', '.')))
                dataset[c] = new_col
            else:
                dataset[c] = data[c]

        dataset.rename(columns={'TOTALMISTAKES': 'mistakes'}, inplace=True)
        dataset.rename(columns={'SEX': 'sex'}, inplace=True)
        dataset.loc[dataset['sex'] == 2, 'sex'] = 0

        # computing the binary outcome
        dataset['mistakes_binary'] = dataset['mistakes'] >= 8
        dataset['mistakes_binary'] = dataset['mistakes_binary'].astype(int)

        dataset.reset_index(drop=True, inplace=True)

        self._ds = dataset

        self._explanatory_variables = ['HEIGHT', 'WEIGHT', 'B-CURLS', '6MInWk', '6STEPS', '10METRE',
                                       'TIMEDUP&GO', '5SittoStands', 'SITTOSTAND', 'GRIPLEFT',
                                       'GRIPRIGHT', 'FR1', 'FR2', 'FR3', 'sex']

        if self.outcome_type == 'binary':
            self._outcome_label = 'mistakes_binary'  # [0, 1]
        else:
            self._outcome_label = 'mistakes'

        self._favorable_label_binary = [0]
        self._favorable_label_continuous = lambda x: x < 8  # 2nd quartile

        # SEX Female = 0, Male = 1
        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # male
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = 'mistakes'
        self._binary_label_name = 'mistakes_binary'

        self._cut_point = 8  # is >= all above is 1
        self._non_favorable_label_continuous = lambda x: x >= 8

        self._fav_dict = None
        self._nonf_dict = None

        self._middle_fav = None
        self._middle_non_fav = None

