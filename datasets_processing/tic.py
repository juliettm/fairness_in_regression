import pandas as pd

from datasets_processing._dataset import BaseDataset, get_the_middle


class TicDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'ordinal'
        self._name = 'tic'
        self._att = att
        self.preprocess()

    def preprocess(self):
        variable_names = ['MOSTYPE', 'MAANTHUI', 'MGEMOMV', 'MGEMLEEF', 'MOSHOOFD', 'MGODRK', 'MGODPR', 'MGODOV',
                          'MGODGE', 'MRELGE', 'MRELSA', 'MRELOV', 'MFALLEEN', 'MFGEKIND', 'MFWEKIND', 'MOPLHOOG',
                          'MOPLMIDD', 'MOPLLAAG', 'MBERHOOG', 'MBERZELF', 'MBERBOER', 'MBERMIDD', 'MBERARBG',
                          'MBERARBO', 'MSKA', 'MSKB1', 'MSKB2', 'MSKC', 'MSKD', 'MHHUUR', 'MHKOOP', 'MAUT1', 'MAUT2',
                          'MAUT0', 'MZFONDS', 'MZPART', 'MINKM30', 'MINK3045', 'MINK4575', 'MINK7512', 'MINK123M',
                          'MINKGEM', 'MKOOPKLA', 'PWAPART', 'PWABEDR', 'PWALAND', 'PPERSAUT', 'PBESAUT', 'PMOTSCO',
                          'PVRAAUT', 'PAANHANG', 'PTRACTOR', 'PWERKT', 'PBROM', 'PLEVEN', 'PPERSONG', 'PGEZONG',
                          'PWAOREG', 'PBRAND', 'PZEILPL', 'PPLEZIER', 'PFIETS', 'PINBOED', 'PBYSTAND', 'AWAPART',
                          'AWABEDR', 'AWALAND', 'APERSAUT', 'ABESAUT', 'AMOTSCO', 'AVRAAUT', 'AAANHANG', 'ATRACTOR',
                          'AWERKT', 'ABROM', 'ALEVEN', 'APERSONG', 'AGEZONG', 'AWAOREG', 'ABRAND', 'AZEILPL',
                          'APLEZIER', 'AFIETS', 'AINBOED', 'ABYSTAND', 'CARAVAN']

        df = pd.read_csv("datasets/tic/ticdata2000.txt", sep='\t',
                         names=variable_names)

        variable_names_tst = ['MOSTYPE', 'MAANTHUI', 'MGEMOMV', 'MGEMLEEF', 'MOSHOOFD', 'MGODRK', 'MGODPR', 'MGODOV',
                              'MGODGE', 'MRELGE', 'MRELSA', 'MRELOV', 'MFALLEEN', 'MFGEKIND', 'MFWEKIND', 'MOPLHOOG',
                              'MOPLMIDD', 'MOPLLAAG', 'MBERHOOG', 'MBERZELF', 'MBERBOER', 'MBERMIDD', 'MBERARBG',
                              'MBERARBO', 'MSKA', 'MSKB1', 'MSKB2', 'MSKC', 'MSKD', 'MHHUUR', 'MHKOOP', 'MAUT1',
                              'MAUT2', 'MAUT0', 'MZFONDS', 'MZPART', 'MINKM30', 'MINK3045', 'MINK4575', 'MINK7512',
                              'MINK123M', 'MINKGEM', 'MKOOPKLA', 'PWAPART', 'PWABEDR', 'PWALAND', 'PPERSAUT', 'PBESAUT',
                              'PMOTSCO', 'PVRAAUT', 'PAANHANG', 'PTRACTOR', 'PWERKT', 'PBROM', 'PLEVEN', 'PPERSONG',
                              'PGEZONG', 'PWAOREG', 'PBRAND', 'PZEILPL', 'PPLEZIER', 'PFIETS', 'PINBOED', 'PBYSTAND',
                              'AWAPART', 'AWABEDR', 'AWALAND', 'APERSAUT', 'ABESAUT', 'AMOTSCO', 'AVRAAUT', 'AAANHANG',
                              'ATRACTOR', 'AWERKT', 'ABROM', 'ALEVEN', 'APERSONG', 'AGEZONG', 'AWAOREG', 'ABRAND',
                              'AZEILPL', 'APLEZIER', 'AFIETS', 'AINBOED', 'ABYSTAND']
        df_tst = pd.read_csv("datasets/tic/ticeval2000.txt", sep='\t',
                             names=variable_names_tst)

        var_obj = ['CARAVAN']
        df_obj = pd.read_csv("datasets/tic/tictgts2000.txt", sep='\t', names=var_obj)

        df_tst_all = pd.concat([df_tst, df_obj], axis=1)
        dataset = pd.concat([df, df_tst_all])
        dataset.reset_index(inplace=True)
        dataset.drop('index', axis=1, inplace=True)

        variable_names_data = ['MOSTYPE', 'MAANTHUI', 'MGEMOMV', 'MGEMLEEF', 'MOSHOOFD', 'MGODRK', 'MGODPR', 'MGODOV',
                               'MGODGE', 'MRELGE', 'MRELSA', 'MRELOV', 'MFALLEEN', 'MFGEKIND', 'MFWEKIND', 'MOPLHOOG',
                               'MOPLMIDD', 'MOPLLAAG', 'MBERHOOG', 'MBERZELF', 'MBERBOER', 'MBERMIDD', 'MBERARBG',
                               'MBERARBO', 'MSKA', 'MSKB1', 'MSKB2', 'MSKC', 'MSKD', 'MHHUUR', 'MHKOOP', 'MAUT1',
                               'MAUT2', 'MAUT0', 'MZFONDS', 'MZPART', 'MINKM30', 'MINK3045', 'MINK4575', 'MINK7512',
                               'MINK123M', 'MINKGEM', 'MKOOPKLA']
        dataset = dataset[variable_names_data]

        # religion
        # 6 MGODRK Roman catholic see L3
        # 7 MGODPR Protestant ...
        # 8 MGODOV Other religion
        # 9 MGODGE No religion
        max_religion = dataset[['MGODRK', 'MGODPR', 'MGODOV', 'MGODGE']].idxmax(axis=1)
        dataset['religion'] = max_religion
        dataset['religion'] = dataset['religion'].apply(lambda x: 0 if x == 'MGODGE' else 1)
        dataset.drop(['MGODRK', 'MGODPR', 'MGODOV', 'MGODGE'], axis=1, inplace=True)

        # income
        # 37 MINKM30 Income < 30.000
        # 38 MINK3045 Income 30-45.000
        # 39 MINK4575 Income 45-75.000
        # 40 MINK7512 Income 75-122.000
        # 41 MINK123M Income >123.000
        # 42 MINKGEM Average income
        # 43 MKOOPKLA Purchasing power class

        # data range
        # 0 - 0 %
        # 1 - 1 - 10 %
        # 2 - 11 - 23 %
        # 3 - 24 - 36 %
        # 4 - 37 - 49 %
        # 5 - 50 - 62 %
        # 6 - 63 - 75 %
        # 7 - 76 - 88 %
        # 8 - 89 - 99 %
        # 9 - 100 %
        dataset['income'] = dataset[['MINKM30', 'MINK3045', 'MINK4575', 'MINK7512', 'MINK123M']].max(axis=1)
        dataset.drop(['MINKM30', 'MINK3045', 'MINK4575', 'MINK7512', 'MINK123M', 'MINKGEM', 'MKOOPKLA'], axis=1,
                     inplace=True)

        # computing the binary outcome
        dataset['income_binary'] = dataset['income'] > 6
        dataset['income_binary'] = dataset['income_binary'].astype(int)

        # income distribution
        # 2       1
        # 3     778
        # 4    2763
        # 5    3462
        # 6    1427
        # 7     771
        # 8     240
        # 9     380

        # delete income 2 case there is only one case
        dataset = dataset.loc[dataset['income'] > 2]

        dataset.reset_index(drop=True, inplace=True)

        self._ds = dataset

        self._explanatory_variables = ['MOSTYPE', 'MAANTHUI', 'MGEMOMV', 'MGEMLEEF', 'MOSHOOFD',
                                       'MRELGE', 'MRELSA', 'MRELOV', 'MFALLEEN', 'MFGEKIND', 'MFWEKIND', 'MOPLHOOG',
                                       'MOPLMIDD', 'MOPLLAAG', 'MBERHOOG', 'MBERZELF', 'MBERBOER', 'MBERMIDD',
                                       'MBERARBG',
                                       'MBERARBO', 'MSKA', 'MSKB1', 'MSKB2', 'MSKC', 'MSKD', 'MHHUUR', 'MHKOOP',
                                       'MAUT1',
                                       'MAUT2', 'MAUT0', 'MZFONDS', 'MZPART', 'religion']

        if self.outcome_type == 'binary':
            self._outcome_label = 'income_binary'  # [0, 1]
        else:
            self._outcome_label = 'income'  # [6-9]

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [7, 8, 9]

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # No religion
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = 'income'
        self._binary_label_name = 'income_binary'

        self._cut_point = 7  # is >= all above is 1
        self._non_favorable_label_continuous = [3, 4, 5, 6]

        self._fav_dict = self.assign_ranges_to_ordinal(fav=True)
        self._nonf_dict = self.assign_ranges_to_ordinal(fav=False, sort_desc=True)

        self._middle_fav = get_the_middle(self.favorable_label_continuous)
        self._middle_non_fav = get_the_middle(self.non_favorable_label_continuous)

