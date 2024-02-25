import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

from datasets_processing._dataset import BaseDataset


class LsatDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'countinuous'
        self._name = 'lsat'
        self._att = att
        self.preprocess()

    def preprocess(self):
        df = pd.read_csv("datasets/lsat/bar_pass_prediction.csv", sep=',')

        data = df[['race1', 'gender', 'age', 'fam_inc', 'fulltime', 'zgpa', 'ugpa', 'lsat']]
        data = data.dropna(axis=0).reset_index()

        race_map = {'asian': 0,
                    'black': 1,
                    'hisp': 2,
                    'other': 3,
                    'white': 4}
        race = [race_map[r] for r in data['race1']]

        # gender 'gender'
        gender_map = {'male': 0, 'female': 1}
        gender = [gender_map[r] for r in data['gender']]

        cat_vars = ['fulltime']
        num_vars = ['fam_inc', 'ugpa', 'lsat']
        df_num = data[num_vars]
        df_num.reset_index(drop=True, inplace=True)

        cat_encoder = OneHotEncoder(sparse=False, drop='first')

        cat_data = cat_encoder.fit_transform(data[cat_vars])

        catnewcols = np.concatenate(
            [[cat] if len(item) == 2 else [cat + '_' + cn for cn in item[1:]] for cat, item in
             zip(cat_vars, cat_encoder.categories_)]).tolist()

        cat_df = pd.DataFrame(cat_data, columns=catnewcols)
        cat_df.reset_index(drop=True, inplace=True)


        object_columns = data.select_dtypes(include=['object']).columns
        data[object_columns] = data[object_columns].apply(pd.to_numeric, errors='coerce',
                                                          downcast='float')

        columns_with_negatives = [col for col in data.columns if data[col].lt(0).any()]
        x = data[columns_with_negatives].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_pos = pd.DataFrame(x_scaled, columns=columns_with_negatives)
        df_pos.reset_index(drop=True, inplace=True)



        dataset = pd.concat([df_num, cat_df, df_pos], axis=1)
        dataset['race'] = race
        dataset['race'] = dataset['race'].apply(lambda x: 1 if x == 4 else 0)
        dataset['gender'] = gender

        dataset.reset_index(drop=True, inplace=True)

        # computing the binary outcome
        # TODO qué es un lsat bueno? 37 is the mean 2nd quartile
        # dataset['lsat_binary'] = dataset['lsat'] >= 37
        # dataset['lsat_binary'] = dataset['lsat_binary'].astype(int)

        upga = dataset['ugpa']
        dataset.drop(['ugpa'], axis=1, inplace=True)
        dataset['ugpa'] = upga
        # computing the binary outcome
        dataset['ugpa_binary'] = dataset['ugpa'] >= 3.2
        dataset['ugpa_binary'] = dataset['ugpa_binary'].astype(int)

        dataset.reset_index(drop=True, inplace=True)

        self._ds = dataset

        # 'ugpa'
        self._explanatory_variables = ['age', 'fam_inc', 'zgpa', 'gender', 'fulltime', 'race']

        if self.outcome_type == 'binary':
            self._outcome_label = 'ugpa_binary'  # [0, 1]
        else:
            self._outcome_label = 'ugpa'

        self._favorable_label_binary = [1]
        # self._favorable_label_continuous = lambda x: x >= 37
        self._favorable_label_continuous = lambda x: x >= 3.2

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # white
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = 'ugpa'
        self._binary_label_name = 'ugpa_binary'

        # self._cut_point = 37  # is >= all above is 1
        # self._non_favorable_label_continuous = lambda x: x < 37
        self._cut_point = 3.2  # is >= all above is 1
        self._non_favorable_label_continuous = lambda x: x < 3.2

        self._fav_dict = None
        self._nonf_dict = None

        self._middle_fav = None
        self._middle_non_fav = None

