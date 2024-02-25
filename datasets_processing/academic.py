# preprocesing
import pandas as pd
from datasets_processing._dataset import BaseDataset, get_the_middle
import numpy as np

# The dataset tried to find the end semester percentage prediction based on different social, economic and academic attributes.

# @ATTRIBUTE ge  {M,F}
# @ATTRIBUTE cst {G,ST,SC,OBC,MOBC}
# @ATTRIBUTE tnp {Best,Vg,Good,Pass,Fail}
# @ATTRIBUTE twp {Best,Vg,Good,Pass,Fail}
# @ATTRIBUTE iap {Best,Vg,Good,Pass,Fail}
# @ATTRIBUTE esp {Best,Vg,Good,Pass,Fail}
# @ATTRIBUTE arr {Y,N}
# @ATTRIBUTE ms  {Married,Unmarried}
# @ATTRIBUTE ls  {T,V}
# @ATTRIBUTE as  {Free,Paid}
# @ATTRIBUTE fmi {Vh,High,Am,Medium,Low}
# @ATTRIBUTE fs  {Large,Average,Small}
# @ATTRIBUTE fq  {Il,Um,10,12,Degree,Pg}
# @ATTRIBUTE mq  {Il,Um,10,12,Degree,Pg}
# @ATTRIBUTE fo  {Service,Business,Retired,Farmer,Others}
# @ATTRIBUTE mo  {Service,Business,Retired,Housewife,Others}
# @ATTRIBUTE nf  {Large,Average,Small}
# @ATTRIBUTE sh  {Good,Average,Poor}
# @ATTRIBUTE ss  {Govt,Private}
# @ATTRIBUTE me  {Eng,Asm,Hin,Ben}
# @ATTRIBUTE tt  {Large,Average,Small}
# @ATTRIBUTE atd {Good,Average,Poor}

column_mapping = {
'ge' : {'M': 0,'F': 1},
'cst' : {'G':1,'ST':2,'SC':3,'OBC':4,'MOBC':4},
'tnp' : {'Best':5, 'Vg':4,'Good':3,'Pass':2,'Fail':1},
'twp'	: {'Best':5,'Vg':4,'Good':3,'Pass':2,'Fail':1},
'iap'	: {'Best':5,'Vg':4,'Good':3,'Pass':2,'Fail':1},
'esp'	: {'Best':5,'Vg':4,'Good':3,'Pass':2,'Fail':1},
'arr'	: {'Y':1,'N':0},
'ms'	: {'Married':0,'Unmarried':1},
'ls'	: {'T':0,'V':1},
'as_v' : {'Free':1,'Paid':0},
'fmi'	: {'Vh':5,'High':4,'Am':3,'Medium':2,'Low':1},
'fs'	: {'Large':3,'Average':2,'Small':1},
'fq'	: {'Il':1,'Um':2,'10':3,'12':4,'Degree':5,'Pg':6},
'mq'	: {'Il':1,'Um':2,'10':3,'12':4,'Degree':5,'Pg':6},
'fo'	: {'Service':1,'Business':2,'Retired':3,'Farmer':4,'Others':5},
'mo'	: {'Service':1,'Business':2,'Retired':3,'Farmer':4,'Others':5},
'nf'	: {'Large':3,'Average':2,'Small':1},
'sh'	: {'Good':3,'Average':2,'Poor':1},
'ss'	: {'Govt':1,'Private':0},
'me'	: {'Eng':1,'Asm':2,'Hin':3,'Ben':4},
'tt'	: {'Large':3,'Average':2,'Small':1},
'atd'	: {'Good':1,'Average':0,'Poor':0}}


class AcademicDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'academic'
        self._att = att
        self.preprocess()

    def preprocess(self):

        column_names = ['ge', 'cst', 'tnp', 'twp', 'iap', 'esp', 'arr', 'ms', 'ls', 'as_v', 'fmi', 'fs', 'fq', 'mq',
                        'fo', 'mo', 'nf', 'sh', 'ss', 'me', 'tt', 'atd']

        data = pd.read_csv("datasets/academic/academic-perf.csv",
                           sep=',', names=column_names)

        df_base = pd.DataFrame()
        # Convert columns to numeric, if possible
        for col, mapping in column_mapping.items():
            df_base[col] = data[col].map(mapping)

        df_base = df_base.dropna(subset = ['ge', 'ms', 'atd'])
        # substitute nan with the mean
        df_base = df_base.apply(lambda x: x.fillna(x.mean()), axis=0)

        # this is always binary
        df_base['binary_atd'] = df_base['atd']

        # Drop columns with only one unique value
        df_base = df_base.loc[:, df_base.nunique() != 1]

        target_variable_ordinal = 'atd'
        target_variable_binary = 'binary_atd'

        self._ds = df_base

        self._explanatory_variables = [ 'ge', 'cst', 'tnp', 'twp', 'iap', 'esp', 'arr', 'ls', 'as_v', 'fmi',
                                        'fs', 'fq', 'mq', 'fo', 'mo', 'nf', 'sh', 'ss', 'me', 'tt']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        # possible sensitive attributes are ge and ms
        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # 'ge' : {'M': 0,'F': 1}, 'ms' {'Married':0,'Unmarried':1},
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
