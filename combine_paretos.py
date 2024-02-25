import os
import pandas as pd


datasets = ['compas', 'tic', 'wine', 'insurance', 'obesity', 'singles', 'drugs', 'parkinson', 'older-adults', 'crime', 'lsat', 'student']
datasets = {
    "compas": "race",
    "crime": "race",
    "drugs": "Gender",
    "insurance": "sex",
    "lsat": "gender",
    "obesity": "Gender",
    "older-adults": "sex",
    "parkinson": "sex",
    "singles": "sex",
    "student": "sex",
    "tic": "religion",
    "wine": "color"
}


# T1- ['compas', 'singles', 'obesity', 'drugs', 'student', 'insurance', 'parkinson', 'older-adults']
# T2- ['compas', 'singles', 'obesity', 'drugs', 'student', 'older-adults']
# T3- ['compas', 'crime', 'lsat', 'tic', 'wine', 'obesity', 'student', 'singles', 'drugs', 'insurance', 'older-adults']

# output = binary, continuous -> objectives_names = binary
# measures_fairness = ['equal_opportunity_difference', 'false_discovery_rate_difference', 'statistical_parity_difference']

# output = binary, continuous -> objectives_names = regression
# measures_fairness = ['Average_Outcome']
output = 'continuous'
objectives_names = 'binary'
measures_fairness = ['equal_opportunity_difference', 'false_discovery_rate_difference', 'statistical_parity_difference']
# regression measures
measures_fairness = ['Average_Outcome']
output = 'continuous'
objectives_names = 'regression'


for dataset, att in datasets.items():
    for measure_fairness in measures_fairness:
        combined_name = objectives_names + '_' + dataset +'_' + att + '_output_' + output + '_measure_' + measure_fairness
        print(combined_name)

        folder_path = 'results/individuals/'+dataset  # Specify the path to the folder containing CSV files
        output_file = 'results/individuals/output_' + combined_name + '.csv'  # Specify the path for the output concatenated CSV file

        files = os.listdir(folder_path)
        combined_data = pd.DataFrame()

        for file in files:

            if file.startswith('individuals_pareto_'+combined_name):
                print(file)
                file_path = os.path.join(folder_path, file)
                data = pd.read_csv(file_path)  # Add 'header=None' to exclude the header
                combined_data = pd.concat([combined_data, data], ignore_index=True)

        combined_data.to_csv(output_file, index=False)