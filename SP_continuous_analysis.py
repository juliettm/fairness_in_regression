import pandas as pd

# Load the provided CSV file
file_path = 'results/regression_tree/processed_quartiles_results_regression_metrics.csv'
tree_data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure and contents
tree_data.head()

# Load the second provided CSV file for the linear method results
linear_file_path = 'results/regression_linear/processed_quartiles_results_regression_metrics.csv'
linear_data = pd.read_csv(linear_file_path)

# Display the first few rows of the dataframe to understand its structure and contents
linear_data.head()

# Remove the 'method' column from both dataframes
# tree_data_cleaned = tree_data.drop('method', axis=1)
# linear_data_cleaned = linear_data.drop('method', axis=1)

# Renaming columns with '_tree' and '_linear' suffixes
tree_data_renamed = tree_data.add_suffix('_tree')
linear_data_renamed = linear_data.add_suffix('_linear')

# Since the merge keys (dataset and attribute) also got suffixes, we need to rename them back
tree_data_renamed.rename(columns={'dataset_tree': 'dataset', 'attribute_tree': 'attribute', 'method_tree': 'method'}, inplace=True)
linear_data_renamed.rename(columns={'dataset_linear': 'dataset', 'attribute_linear': 'attribute', 'method_linear': 'method'}, inplace=True)

# Merge the dataframes using 'dataset' and 'attribute' as keys
merged_df = pd.merge(tree_data_renamed, linear_data_renamed, on=['dataset', 'attribute', 'method'])

merged_df.head()

# Removing 'accuracies' columns
merged_df = merged_df.drop(['accuracies_tree', 'accuracies_linear'], axis=1)

# Sorting columns as per the specified order: dataset, attribute, then linear values followed by tree values
# and sorting the measures in the order of mae, sp_avg, and statistical_parity
columns_order = [
    'dataset', 'attribute', 'method',
    'mae_n_linear', 'sp_avg_outcome_n_linear', 'statistical_parity_difference_linear',
    'mae_n_tree', 'sp_avg_outcome_n_tree', 'statistical_parity_difference_tree'
]

merged_df_sorted = merged_df[columns_order]

merged_df_sorted.head()

# Removing the 'attribute' column and generating the LaTeX code for the updated dataframe
merged_df_without_attribute = merged_df_sorted.drop('attribute', axis=1)
latex_table_without_attribute = merged_df_without_attribute.to_latex(index=False)

print(latex_table_without_attribute)


# Sorting the datasets in the specified manner: first the specific set in alphabetical order,
# followed by the rest in alphabetical order
specified_datasets = ['compas', 'obesity', 'drugs', 'insurance', 'parkinson', 'older-adults', 'crime']
rest_of_datasets = sorted(set(merged_df_without_attribute['dataset']) - set(specified_datasets))

# Sorting the specified datasets in alphabetical order
specified_sorted = merged_df_without_attribute[merged_df_without_attribute['dataset'].isin(specified_datasets)].sort_values('dataset')

# Sorting the rest of the datasets in alphabetical order
rest_sorted = merged_df_without_attribute[merged_df_without_attribute['dataset'].isin(rest_of_datasets)].sort_values('dataset')

# Combining the two sorted parts
sorted_merged_df = pd.concat([specified_sorted, rest_sorted])

# Generating the LaTeX code for the sorted dataframe
sorted_latex_table = sorted_merged_df.to_latex(index=False, float_format="%.2f", escape=False)

print(sorted_latex_table)

exit(0)

from scipy.stats import wilcoxon

# Extracting the SP DAvgO values for linear and tree methods
sp_avg_linear = sorted_merged_df['sp_avg_outcome_n_linear']
sp_avg_tree = sorted_merged_df['sp_avg_outcome_n_tree']

# Performing the Wilcoxon signed-rank test
wilcoxon_test_result = wilcoxon(sp_avg_linear, sp_avg_tree)

wilcoxon_test_result

