from datasets_processing.compas import CompasDataset
from datasets_processing.crime import CrimeDataset
from datasets_processing.drugs import DrugsDataset
from datasets_processing.insurance import InsuranceDataset
from datasets_processing.lsat import LsatDataset
from datasets_processing.obesity import ObesityDataset
from datasets_processing.older_adults import OlderAdultsDataset
from datasets_processing.parkinson import ParkinsonDataset
from datasets_processing.singles import SinglesDataset
from datasets_processing.student import StudentDataset
from datasets_processing.tic import TicDataset
from datasets_processing.wine import WineDataset


def get_datasets(outcome):
    if outcome == 'binary':
        return [CompasDataset('race', outcome_type=outcome),
                WineDataset('color', outcome_type=outcome),
                SinglesDataset('sex', outcome_type=outcome),
                #TicDataset('religion', outcome_type=outcome),
                ObesityDataset('Gender', outcome_type=outcome),
                DrugsDataset('Gender', outcome_type=outcome),
                # Continuous
                InsuranceDataset('sex', outcome_type=outcome),
                ParkinsonDataset('sex', outcome_type=outcome),
                CrimeDataset('race', outcome_type=outcome),
                OlderAdultsDataset('sex', outcome_type=outcome),
                LsatDataset('race', outcome_type=outcome),
                StudentDataset('sex', outcome_type=outcome)
                ]
    elif outcome == 'ordinal':
        return [CompasDataset('race', outcome_type=outcome),
                WineDataset('color', outcome_type=outcome),
                SinglesDataset('sex', outcome_type=outcome),
                #TicDataset('religion', outcome_type=outcome),
                ObesityDataset('Gender', outcome_type=outcome),
                DrugsDataset('Gender', outcome_type=outcome)
                ]
    elif outcome == 'continuous':
        return [InsuranceDataset('sex', outcome_type=outcome),
                ParkinsonDataset('sex', outcome_type=outcome),
                CrimeDataset('race', outcome_type=outcome),
                OlderAdultsDataset('sex', outcome_type=outcome),
                LsatDataset('race', outcome_type=outcome),
                StudentDataset('sex', outcome_type=outcome)
                ]
    else:
        raise AssertionError('not a valid outcome: ', outcome)