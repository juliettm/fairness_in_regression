import pandas as pd

from datasets_processing._dataset import BaseDataset, get_the_middle
from sklearn import preprocessing


# TODO definir si me quedo con las variables tipo decimal o si cambio como está el resto one hot encoder.
# TODO Elimino las personas que nunca han consumido?

class DrugsDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'ordinal'
        self._name = 'drugs'
        self._att = att
        self.preprocess()

    def preprocess(self):

        Cat = 'cat'
        Num = 'num'

        VARIABLES = {'ID': Cat,  # not used for prediction
                     'Age': Num,
                     'Gender': Cat,
                     'Education': Cat,
                     'Country': Cat,
                     'Race': Cat,
                     'Nscore': Num,
                     'Escore': Num,
                     'Oscore': Num,
                     'Ascore': Num,
                     'Cscore': Num,
                     'Impulsive': Num,
                     'SS': Num,
                     # Drug usage outcomes
                     'Alcohol': Cat,
                     'Amphet': Cat,
                     'Amyl': Cat,
                     'Benzos': Cat,
                     'Caff': Cat,
                     'Cannabis': Cat,
                     'Choc': Cat,
                     'Coke': Cat,
                     'Crack': Cat,
                     'Ecstasy': Cat,
                     'Heroin': Cat,
                     'Ketamine': Cat,
                     'Legalh': Cat,
                     'LSD': Cat,
                     'Meth': Cat,
                     'Mushrooms': Cat,
                     'Nicotine': Cat,
                     'Semer': Cat,
                     'VSA': Cat
                     }

        df = pd.read_csv("datasets/drugs/drug_consumption.data", names=VARIABLES.keys(), sep=',', index_col=0)

        vars = {'Age': Num,
                'Gender': Cat,
                'Education': Cat,
                'Country': Cat,
                'Race': Cat,
                'Nscore': Num,
                'Escore': Num,
                'Oscore': Num,
                'Ascore': Num,
                'Cscore': Num,
                'Impulsive': Num,
                'SS': Num,
                'Coke': Cat}

        class_map = {
            'CL0': 0,
            'CL1': 1,
            'CL2': 2,
            'CL3': 3,
            'CL4': 4,
            'CL5': 5,
            'CL6': 6
        }


        list_cols = ['Age', 'Education', 'Country', 'Race', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']

        x = df[list_cols].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_pos = pd.DataFrame(x_scaled, columns=list_cols)
        df_pos.reset_index(drop=True, inplace=True)

        df_atts = df[['Gender', 'Coke']]
        df_atts.reset_index(drop=True, inplace=True)

        # gender_tr = {
        #     0.48246: 'Female',  # 49.97%
        #     -0.48246: 'Male'}  # 50.03%

        dataset = pd.concat([df_pos, df_atts], axis=1)
        dataset.loc[:, 'Gender'] = (dataset['Gender'] == 0.48246).astype(float)

        # Convert columns to numeric, if possible
        dataset['Coke'] = dataset['Coke'].map(class_map)
        #dataset.loc[:, 'Coke'] = [class_map[item] for item in dataset['Coke']]

        # computing the binary outcome
        dataset['Coke_binary'] = dataset['Coke'] >= 3
        dataset['Coke_binary'] = dataset['Coke_binary'].astype(int)

        dataset.reset_index(drop=True, inplace=True)

        self._ds = dataset

        self._explanatory_variables = ['Age', 'Gender', 'Education', 'Country', 'Race', 'Nscore', 'Escore', 'Oscore',
                                       'Ascore', 'Cscore', 'Impulsive', 'SS']

        if self.outcome_type == 'binary':
            self._outcome_label = 'Coke_binary'  # [0,1]
        else:
            self._outcome_label = 'Coke'  # [0-6]

        self._favorable_label_binary = [0]
        self._favorable_label_continuous = [0, 1, 2]

        self._protected_att_name = ['Gender']
        self._privileged_classes = [[1]]  # female
        self._privileged_groups = [{'Gender': 1}]
        self._unprivileged_groups = [{'Gender': 0}]

        self._continuous_label_name = 'Coke'
        self._binary_label_name = 'Coke_binary'

        self._cut_point = 3  # is >= all above is 1
        self._non_favorable_label_continuous = [3, 4, 5, 6]

        self._fav_dict = self.assign_ranges_to_ordinal(fav=True)
        self._nonf_dict = self.assign_ranges_to_ordinal(fav=False, sort_desc=True)

        self._middle_fav = get_the_middle(self.favorable_label_continuous)
        self._middle_non_fav = get_the_middle(self.non_favorable_label_continuous)
