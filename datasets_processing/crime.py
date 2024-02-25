import numpy as np
import pandas as pd

from datasets_processing._dataset import BaseDataset


# TODO communities and crime and lsat tienen 5 valores en race... dejar esto o hacer como en _compas -- now: 1-White 0-Other
class CrimeDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'countinuous'
        self._name = 'crime'
        self._att = att
        self.preprocess()

    def preprocess(self):

        column_name = ["state", "county", "community", "communityname", "fold", "population", "householdsize",
                       "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29",
                       "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf",
                       "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap",
                       "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov",
                       "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy",
                       "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce",
                       "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
                       "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg",
                       "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig",
                       "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell",
                       "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous",
                       "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR",
                       "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos",
                       "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal",
                       "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc",
                       "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn",
                       "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT",
                       "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq",
                       "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite",
                       "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits",
                       "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars",
                       "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn",
                       "PolicBudgPerPop", "ViolentCrimesPerPop"]

        df = pd.read_csv("datasets/crime/communities.data", sep=',', names=column_name)
        # remove the variables not used for prediction
        data = df.drop(['state', 'county', 'community', 'communityname', 'fold'], axis=1)

        # drop observations with missing values
        data = data.replace('?', np.nan)
        data = data[data.columns[data.isna().mean(0) < 0.1]]
        data = data.replace('?', np.nan).dropna(axis=0, how='any')
        dataset = data.astype(float)

        # race column
        # race -> racepctblack - 0, racePctWhite - 1, racePctAsian - 2, racePctHisp - 3
        race = np.argmax(data[['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']].values, 1)
        dataset['race'] = race
        # 1-White 0-Other
        dataset['race'] = dataset['race'].apply(lambda x: 1 if x == 1 else 0)

        dataset.drop(['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'], axis=1, inplace=True)

        # computing the binary outcome
        dataset['ViolentCrimesPerPop_binary'] = dataset['ViolentCrimesPerPop'] >= 0.15
        dataset['ViolentCrimesPerPop_binary'] = dataset['ViolentCrimesPerPop_binary'].astype(int)

        for e in ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 'ViolentCrimesPerPop', 'state',
                  'county', 'community', 'communityname', 'fold']:
            column_name.remove(e)

        column_name.append('race')
        dataset.reset_index(drop=True, inplace=True)

        self._ds = dataset

        self._explanatory_variables = ['population', 'householdsize', 'agePct12t21', 'agePct12t29',
                                       'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome',
                                       'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst',
                                       'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap',
                                       'indianPerCap', 'AsianPerCap', 'OtherPerCap', 'HispPerCap',
                                       'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad',
                                       'PctBSorMore', 'PctUnemployed', 'PctEmploy', 'PctEmplManu',
                                       'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce',
                                       'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam',
                                       'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par',
                                       'PctWorkMomYoungKids', 'PctWorkMom', 'NumIlleg', 'PctIlleg', 'NumImmig',
                                       'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10',
                                       'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10',
                                       'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam',
                                       'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous',
                                       'PersPerRentOccHous', 'PctPersOwnOccup', 'PctPersDenseHous',
                                       'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup',
                                       'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt',
                                       'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal',
                                       'OwnOccHiQuart', 'RentLowQ', 'RentMedian', 'RentHighQ', 'MedRent',
                                       'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg',
                                       'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState',
                                       'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LandArea',
                                       'PopDens', 'PctUsePubTrans', 'LemasPctOfficDrugUn', 'race']

        if self.outcome_type == 'binary':
            self._outcome_label = 'ViolentCrimesPerPop_binary'  # [0, 1]
        else:
            self._outcome_label = 'ViolentCrimesPerPop'

        self._favorable_label_binary = [0]
        self._favorable_label_continuous = lambda x: x < 0.15

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]  # white
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]
        # Race was replaced to be 1-White 0-Others
        # self._unprivileged_groups = [{'race': 0}, {'race': 2}, {'race': 3}]

        self._continuous_label_name = 'ViolentCrimesPerPop'
        self._binary_label_name = 'ViolentCrimesPerPop_binary'

        self._cut_point = 0.15  # TODO when computing regression this could be an issue is >= all above is 1
        self._non_favorable_label_continuous = lambda x: x >= 0.15

        self._fav_dict = None
        self._nonf_dict = None

        self._middle_fav = None
        self._middle_non_fav = None
