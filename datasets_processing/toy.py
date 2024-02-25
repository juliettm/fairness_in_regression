# preprocesing
import pandas as pd
from sklearn import preprocessing
import numpy as np

from datasets_processing._dataset import BaseDataset, get_the_middle

def generate_toy_data(n_samples, n_samples_low, n_dimensions):
    np.random.seed(0)
    varA = 0.8
    aveApos = [-1.0] * n_dimensions
    aveAneg = [1.0] * n_dimensions
    varB = 0.5
    aveBpos = [0.5] * int(n_dimensions / 2) + [-0.5] * int(n_dimensions / 2 + n_dimensions % 2)
    aveBneg = [0.5] * n_dimensions

    X = np.random.multivariate_normal(aveApos, np.diag([varA] * n_dimensions), n_samples)
    X = np.vstack([X, np.random.multivariate_normal(aveAneg, np.diag([varA] * n_dimensions), n_samples)])
    X = np.vstack([X, np.random.multivariate_normal(aveBpos, np.diag([varB] * n_dimensions), n_samples_low)])
    X = np.vstack([X, np.random.multivariate_normal(aveBneg, np.diag([varB] * n_dimensions), n_samples)])
    sensible_feature = [1] * (n_samples * 2) + [-1] * (n_samples + n_samples_low)
    sensible_feature = np.array(sensible_feature)
    sensible_feature.shape = (len(sensible_feature), 1)
    X = np.hstack([X, sensible_feature])
    y = [1] * n_samples + [-1] * n_samples + [1] * n_samples_low + [-1] * n_samples
    y = np.array(y)
    sensible_feature_id = len(X[1, :]) - 1
    idx_A = list(range(0, n_samples * 2))
    idx_B = list(range(n_samples * 2, n_samples * 3 + n_samples_low))

    # print('(y, sensible_feature):')
    # for el in zip(y, sensible_feature):
    #     print(el)
    return X, y, sensible_feature_id, idx_A, idx_B

class ToyDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'toy'
        self._att = att
        self.preprocess()

    def preprocess(self):

        X, y, sensible_feature_id, idx_A, idx_B = generate_toy_data(10000, 3500, 10)


        df_base = pd.DataFrame(X)
        df_base['Label'] = y
        df_base.loc[df_base['Label'] == -1, 'Label'] = 0
        df_base.rename(columns={10: 'sst'}, inplace=True)

        columns_with_negatives = [col for col in df_base.columns if df_base[col].lt(0).any()]
        rest = df_base.columns.difference(columns_with_negatives)

        x = df_base[columns_with_negatives].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_pos = pd.DataFrame(x_scaled, columns=columns_with_negatives)

        df_rest = df_base[rest]
        df_rest.reset_index(drop=True, inplace=True)

        df_def = pd.concat([df_pos, df_rest], axis=1)


        # this is always binary
        df_def['binary_Label'] = df_base['Label']

        target_variable_ordinal = 'Label'
        target_variable_binary = 'binary_Label'

        self._ds = df_def

        self._explanatory_variables = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'sst']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal

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
