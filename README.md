# General Procedure to Measure Fairness in Regression Problems

The general procedure to measure fairness in regression problems is the following: [General Procedure to Measure Fairness in Regression Problems](plots/Procedure.pdf "General Procedure to Measure Fairness in Regression Problems")

## Datasets


| Output Type  | Dataset       | PA     | PV / UV                              | Variable to Predict      | FL      | Threshold | 
|--------------|---------------|--------|--------------------------------------|--------------------------|---------|-----------|
| Ordinal      | wine          | Color  | **Red - 1** / White - 0              | Quality [3-8]            | [6-8]   | 6         |
|              | COMPAS        | Race   | **White - 1** / AA - 0               | Recidivism [1-10]        | [1-4]   | 5         |
|              | singles       | Gender | **Male - 1** / Female - 0            | Income [1-9]             | [5-9]   | 5         |
|              | obesity       | Gender | **Male - 1** / Female - 0            | Obesity [0-5]            | [0-3]   | 4         |
|              | drugs         | Gender | **Female - 1** / Male - 0            | Coke Recency [0-6]       | [0-2]   | 3         |
| Continuous   | insurance     | Gender | **Male - 1** / Female - 0            | Charges                  | `<= 40K`| 40K       |
|              | parkinson     | Gender | **Male - 1** / Female - 0            | UPDRS score              | `< 17.1`| 17.1      |
|              | older-adults  | Gender | **Male - 1** / Female - 0            | Mistakes                 | `< 8`   | 8         |
|              | crime         | Race   | **White - 1** / Other - 0            | `%` Crimes               | `< 0.15`| 0.15      |
|              | LSAC          | Race   | **White - 1** / Other - 0            | ugpa^1^                  | `>= 3.2`| 3.2       |
|              | student       | Gender | **Female - 1** / Male - 0            | Final Grade              | `>= 12` | 12        |

1: Undergraduate Grade Score Average

## Code

### 1. Discretisation results

The file `discretising_values.py` contains the code for the computation of SP with different thresholds and with the K-means method. 
The figures can be found in the `plots/discretising` folder.
The file `sp_discrete.txt` contains the results of the discretisation of the continuous features in the datasets.

### 2. SP measures computation for binary and continuous outputs - Linear and Tree models

- **Running the experiments:**
The file `experiment_linear.py` contains the code for the computation of SP in all datasets using the linear model.
The file `experiment_tree.py` contains the code for the computation of SP in all datasets using the tree model.
Each experiment should be run with the values 'ordinal', 'binary' and 'continuous' for the output type.
The results are in the folder `results/regression_linear` and `results/regression_tree` respectively.


- **Processing the results:**
The files `process_quartiles_regression_spc.py` and `process_quartiles_regression_spr.py` are used to compute the files just with the mean values for each dataset in each folder: `processed_quartiles_results_classification_metrics.csv` and `processed_quartiles_results_regression_metrics.csv`.


- **Plotting the results for SP classification measure:**
`SP_comparison_plots.ipynb` contains the code to generate the plots comparing the SP measures for the linear and tree models. The plot can be found in the `plots` folder as SP_linear_comparison.pdf and SP_tree_comparison.pdf.


- **SP measure for continuous output table:**
`SP_continuous_analysis.py` contains the code to generate the table with the SP measures for the continuous output. 


- **Plotting the results for SP continuos measure:**
`SP_comparison_plots.ipynb` contains the code to generate the plots comparing the SP measures for the tree model. The plot can be found in the `plots` folder as SP_continuous_band.pdf.

### 3. Hyper-parameter optimisation

- **Running the experiments:**
The experiments for the hyper-parameter optimisation are a modification of the implementation that can be found in the repository: [https://github.com/anavaldi/fairness_nsga] to use the multi-objective parameter optimisation in regression setting.
The results are in the folder `results/individuals/`.

- **Processing the results:**
We use `combine_paretos.py` to combine all the solutions in a single file. The result files can be found in the same folder. 
This file is used in `plots.py` to generate the paretos figures of the results saved in the folder `plots`. 
