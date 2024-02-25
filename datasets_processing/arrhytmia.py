# preprocesing
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing


from datasets_processing._dataset import BaseDataset, get_the_middle

COLNAMES = ['sex', 'height', 'age', 'weight',
            'QRS duration', 'P-R interval', 'Q-T interval', 'T interval',
            'P interval', 'QRS', 'T', 'P',
            'QRST', 'J', 'heart rate',
            'Q wave', 'R wave', 'S wave', 'R` wave', 'S` wave',
            'Number of intrinsice deflections',
            'Existence of ragged R wave',
            'Existence of diphasic derivation of R wave',
            'Existence of ragged P wave',
            'Existence of diphasic derivation of P wave',
            'Existence of ragged T wave',
            'Existence of diphasic derivation of T wave']

COLNAMES += [f'V{id:03d}' for id in range(len(COLNAMES) + 1, 279)]
COLNAMES += ['class']

Cat = 'categorical'
Num = 'numeric'
VARTYPES = [Num] * len(COLNAMES)
for i in [0] + list(range(21, 27)) + [278]:
    VARTYPES[i] = Cat



class ArrhythmiaDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'arrhythmia'
        self._att = att
        self.preprocess()

    def preprocess(self):

        data = pd.read_csv("datasets/arrhythmia/arrhythmia.data",
                           sep=',', names=COLNAMES)

        # drop two unordinary samples with 500+ heights
        data = data[data['height'] < 500]

        # Remove the existence of ragged/disphasic derviation waves variables
        _COLNAMES = np.delete(COLNAMES, [12] + list(range(21, 27)))
        _VARTYPES = np.delete(VARTYPES, [12] + list(range(21, 27)))

        data = data[_COLNAMES]

        # ? -> np.nan, drop na
        data = data.replace('?', np.nan).dropna(axis=0, how='any')

        # Compile binary classification problem: normal vs. others
        class_original = data['class']
        target = (data['class'] == 1).astype(float)
        gender = data['sex'] # (0 = male; 1 = female)

        target.reset_index(drop=True, inplace=True)
        gender.reset_index(drop=True, inplace=True)

        data.drop(['class', 'sex'], axis=1, inplace=True)

        # remove zero-variance features
        selector = VarianceThreshold(0)
        data = pd.DataFrame(selector.fit_transform(data), index=data.index,
                            columns=data.columns[selector.get_support()])

        # Select only object columns
        object_columns = data.select_dtypes(include=['object']).columns

        # Convert all object columns to float type, errors set to 'coerce' will replace non-convertible values with NaN
        data[object_columns] = data[object_columns].apply(pd.to_numeric, errors='coerce',
                                                                                downcast='float')

        columns_with_negatives = [col for col in data.columns if data[col].lt(0).any()]
        rest = data.columns.difference(columns_with_negatives)

        x = data[columns_with_negatives].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_pos = pd.DataFrame(x_scaled, columns=columns_with_negatives)

        df_rest = data[rest]
        df_rest.reset_index(drop=True, inplace=True)

        df_def = pd.concat([df_rest, df_pos], axis=1)
        df_def['sex'] = gender
        df_def['arrhythmia'] = target

        # this is always binary
        df_def['binary_arrhythmia'] = df_def['arrhythmia']

        target_variable_ordinal = 'arrhythmia'
        target_variable_binary = 'binary_arrhythmia'


        self._ds = df_def

        self._explanatory_variables = [ 'height', 'age', 'weight', 'QRS duration', 'P-R interval', 'Q-T interval',
                                        'T interval', 'P interval', 'QRS', 'T', 'P', 'J', 'heart rate', 'Q wave',
                                        'R wave', 'S wave', 'S` wave', 'V028', 'V029', 'V030', 'V031', 'V032', 'V033',
                                        'V034', 'V035', 'V036', 'V037', 'V038', 'V039', 'V040', 'V041', 'V042', 'V043',
                                        'V044', 'V045', 'V046', 'V047', 'V048', 'V049', 'V051', 'V052', 'V053', 'V054',
                                        'V055', 'V056', 'V057', 'V058', 'V059', 'V060', 'V061', 'V062', 'V063', 'V064',
                                        'V065', 'V066', 'V068', 'V070', 'V072', 'V073', 'V075', 'V076', 'V077', 'V078',
                                        'V079', 'V080', 'V081', 'V082', 'V085', 'V086', 'V087', 'V088', 'V089', 'V090',
                                        'V091', 'V092', 'V093', 'V094', 'V095', 'V096', 'V097', 'V098', 'V099', 'V100',
                                        'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110',
                                        'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120',
                                        'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130',
                                        'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V140', 'V142', 'V144', 'V146',
                                        'V147', 'V148', 'V149', 'V150', 'V152', 'V153', 'V154', 'V155', 'V158', 'V159',
                                        'V160', 'V161', 'V162', 'V163', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170',
                                        'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180',
                                        'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190',
                                        'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199', 'V200',
                                        'V201', 'V202', 'V203', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211',
                                        'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221',
                                        'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V230', 'V231',
                                        'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V240', 'V241',
                                        'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251',
                                        'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261',
                                        'V262', 'V263', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270', 'V271', 'V272',
                                        'V273', 'V275', 'V276', 'V277', 'V278', 'sex']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal  # [1-9]

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # (0 = male; 1 = female)
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

ArrhythmiaDataset('kk')

