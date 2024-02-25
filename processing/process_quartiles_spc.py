import os
import glob
import pandas as pd

# Define the folder path where the files are located
folder_path = '../results/regression_tree'  # Update this to your specific folder path

# Get all file paths that end with '_quartiles.csv'
quartiles_csv_files = glob.glob(os.path.join(folder_path, '*_quartiles.csv'))

# Initialize an empty DataFrame to collect all the results
all_results = pd.DataFrame()

# Process each file
for file in quartiles_csv_files:
    print(file)
    # Read the csv file into a DataFrame
    df = pd.read_csv(file)

    # Set 'Unnamed: 0' as the index
    df.set_index('Unnamed: 0', inplace=True)

    # Keep only the mean row for the specified columns and round to 3 decimal places
    mean_data = df.loc['mean', ['accuracies', 'statistical_parity_difference',
                                'equal_opportunity_difference', 'false_discovery_rate_difference', 'sp_avg_outcome_n']].round(3)

    # Extract the first three elements of the filename (separated by '_')
    file_name_elements = file.split('/')[-1].split('_')[:4]

    # Add each element as a separate column to the DataFrame
    mean_data = mean_data.to_frame().transpose()
    mean_data['dataset'] = file_name_elements[0]

    # Attributes for the linear results
    # mean_data['attribute'] = file_name_elements[3]
    # mean_data['method'] = file_name_elements[1]

    # Attributes for the tree results
    mean_data['attribute'] = file_name_elements[1]
    mean_data['method'] = file_name_elements[2]

    # Append the results
    all_results = pd.concat([all_results, mean_data])

# Save all the results to a CSV file
output_file_path = os.path.join(folder_path, 'processed_quartiles_results_classification_metrics.csv')
all_results.to_csv(output_file_path, index=False)
