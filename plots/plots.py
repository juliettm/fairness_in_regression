# -*- coding: utf-8 -*-
# change the name of the experiment EX-TX
# change the name of the dataset
# change the name of the output and the rest of the parameters (if needed)


import pandas as pd
import numpy as np

pareto_metrics_dataframe = pd.DataFrame()

# Solo activar cuando se hacen los E1 --- en plots poner una condicion para calcular las métricas del pareto.
pareto_metrics = False

def print_pareto_measures(df):
    # Printing pareto measures, X -> fairness measure Y -> error
    # Sort DataFrame by 'Y' in descending order for Pareto analysis
    df_sorted = df.sort_values(by='Y', ascending=False)

    # Calculate cumulative sum and percentage for 'Y'
    df_sorted['Cumulative_Sum'] = df_sorted['Y'].cumsum()
    df_sorted['Cumulative_Percentage'] = 100 * df_sorted['Cumulative_Sum'] / df_sorted['Y'].sum()

    # 1. Slope between first and last point
    # (y2 - y1) / (x2 - x1)
    first_point = df_sorted['Cumulative_Percentage'].iloc[0]
    last_point = df_sorted['Cumulative_Percentage'].iloc[-1]
    slope_cum_p = (last_point - first_point) / (df_sorted.shape[0] - 1)
    # Slope for each point with the next
    slope = (df_sorted['Y'].iloc[-1] - df_sorted['Y'].iloc[0]) / (df_sorted['X'].iloc[-1] - df_sorted['X'].iloc[0])
    list_slope = []
    for i in range(df_sorted.shape[0] - 1):
        list_slope.append((df_sorted['Y'].iloc[i + 1] - df_sorted['Y'].iloc[i]) / (df_sorted['X'].iloc[i + 1] - df_sorted['X'].iloc[i]))
    list_slope.append(0)
    df_sorted['Slope'] = list_slope


    # 2. Chirality: Check for 'flat' regions by looking at the difference between adjacent cumulative percentages
    df_sorted['Percentage_Diff'] = df_sorted['Cumulative_Percentage'].diff().fillna(0)
    chirality_points = df_sorted[df_sorted['Slope'].between(-1.0, 1.0)]['X'].tolist()

    # Create an empty list to hold chirality subsets
    chirality_subsets = []

    print(df_sorted)

    # Initialize a temporary subset
    temp_subset = []

    # Iterate through the DataFrame to find 'flat' regions (chirality points)
    for index, row in df_sorted.iterrows():
        if -1 < row['Slope'] < 1:  # Replace 5 with your threshold for a 'flat' region --Percentage_Diff
            temp_subset.append(row['X'])
        else:
            if len(temp_subset) > 1:
                chirality_subsets.append(temp_subset)
            temp_subset = []

    # Append any remaining subset
    if len(temp_subset) > 1:
        chirality_subsets.append(temp_subset)

    # 3. Min and Max Values
    min_value_error = df_sorted['Y'].min()
    max_value_error = df_sorted['Y'].max()
    min_value_measure = df_sorted['X'].min()
    max_value_measure = df_sorted['X'].max()


    # 4. Variability: Standard Deviation
    std_dev = np.std(df_sorted['Y'])

    pareto_measures = pd.DataFrame({
    "Slope:": slope,
    "Totql Points": df_sorted.shape[0],
    "Chirality Points:": [chirality_points],
    "Number Chirality Points:": len(chirality_points),
    "Chirality Subsets:": [chirality_subsets],
    "Number Chirality Subsets:": len(chirality_subsets),
    "Min Value E:": min_value_error,
    "Max Value E:": max_value_error,
    "Min Value M:": min_value_measure,
    "Max Value M:": max_value_measure,
    "Standard Deviation:": std_dev})

    return pareto_measures

# TODO Needs to be done one by one now
datasets = ['lsat_gender']
datasets = ['lsat']


# E1-T1 datasets = ['compas', 'singles', 'drugs', 'parkinson', 'older-adults', 'crime', 'lsat', 'student']
# E1-T2 datasets = ['compas', 'singles', 'drugs', 'parkinson', 'older-adults', 'crime', 'lsat', 'student']

# E3-SP-T3T4 datasets = ['compas', 'singles', 'drugs', 'parkinson', 'older-adults', 'crime', 'lsat', 'student']



for dataset in datasets:

    # Path to the combined parteos files
    dsname = {
        'dataset_eo_dt_bb': '../results/individuals/output_binary_{}_output_binary_measure_equal_opportunity_difference.csv'.format(dataset),
        'dataset_fdr_dt_bb': '../results/individuals/output_binary_{}_output_binary_measure_false_discovery_rate_difference.csv'.format(dataset),
        'dataset_sp_dt_bb': '../results/individuals/output_binary_{}_output_binary_measure_statistical_parity_difference.csv'.format(dataset),
        'dataset_eo_dt_cb': '../results/individuals/output_binary_{}_output_continuous_measure_equal_opportunity_difference.csv'.format(dataset),
        'dataset_fdr_dt_cb': '../results/individuals/output_binary_{}_output_continuous_measure_false_discovery_rate_difference.csv'.format(dataset),
        'dataset_sp_dt_cb': '../results/individuals/output_binary_{}_output_continuous_measure_statistical_parity_difference.csv'.format(dataset),
        'dataset_sp_dt_rb': '../results/individuals/output_regression_{}_output_binary_measure_Average_Outcome.csv'.format(dataset),
        'dataset_sp_dt_rc': '../results/individuals/output_regression_{}_output_continuous_measure_Average_Outcome.csv'.format(dataset)
    }


    titles = {
        'dataset_eo_dt_bb': dataset,
        'dataset_fdr_dt_bb': dataset,
        'dataset_sp_dt_bb': dataset,
        'dataset_eo_dt_cb': dataset,
        'dataset_fdr_dt_cb': dataset,
        'dataset_sp_dt_cb': dataset,
        'dataset_sp_dt_rb': dataset,
        'dataset_sp_dt_rc': dataset
    }

    # Experiment 1: compare the three fairness measures for the datasets

    # TODO Falta E1-T3 y E1-T4 (regression measures to be done)

    # name_experiment = 'E1-T2'
    # problems = ['dataset_eo_dt_cb', 'dataset_fdr_dt_cb', 'dataset_sp_dt_cb']

    # name_experiment = 'E1-T1'
    # problems = ['dataset_eo_dt_bb', 'dataset_fdr_dt_bb', 'dataset_sp_dt_bb']


    # Experiment 3: compare the same fairness measure for the datasets with same measure to be optimised type
    # TODO Falta E3-EO-T3T4 and E3-FDR-T3T4 (regression measures to be done)
    name_experiment = 'E3-SP-T1T2'
    problems = ['dataset_sp_dt_bb', 'dataset_sp_dt_cb']
    ylabel = '1$-$G-mean'
    #
    # name_experiment = 'E3-EO-T1T2'
    # problems = ['dataset_eo_dt_bb', 'dataset_eo_dt_cb']
    #
    # name_experiment = 'E3-FDR-T1T2'
    # problems = ['dataset_fdr_dt_bb', 'dataset_fdr_dt_cb']
    #
    # name_experiment = 'E3-SP-T3T4'
    # problems = ['dataset_sp_dt_rb', 'dataset_sp_dt_rc']
    # ylabel = 'MSE'


    prefix = None
    #ylabel = '1$-$G-mean'

    prefix = [' with $f_1(\hat{y}, y) = 1 - \mathrm{G}$-$\mathrm{mean} = 1 - \sqrt{TPR \cdot TNR}$',' with $f_1(\hat{y}, y) = 1 - \mathrm{F}_1$-$\mathrm{score}=1 - \\frac{2}{TPR^{-1}+PPV^{-1}}$']
    #ylabel = '$f(\hat{y},y)$ (error)'



    print_params = False
    combine_plots = True

# TODO cambiar esto si no se quieren las sombras
    # if combine_plots:
    #    plot_shadow = False
    # else:
    plot_shadow = True

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np



    #------------------------------------------------------------------------------
    def makeTable(headerRow,columnizedData,columnSpacing=2):
        """Creates a technical paper style, left justified table

        Author: Christopher Collett
        Date: 6/1/2019"""
        from numpy import array,max,vectorize

        cols = array(columnizedData,dtype=str)
        colSizes = [max(vectorize(len)(col)) for col in cols]

        header = ''
        rows = ['' for i in cols[0]]

        for i in range(0,len(headerRow)):
            if len(headerRow[i]) > colSizes[i]: colSizes[i]=len(headerRow[i])
            headerRow[i]+=' '*(colSizes[i]-len(headerRow[i]))
            header+=headerRow[i]
            if not i == len(headerRow)-1: header+=' '*columnSpacing

            for j in range(0,len(cols[i])):
                if len(cols[i][j]) < colSizes[i]:
                    cols[i][j]+=' '*(colSizes[i]-len(cols[i][j])+columnSpacing)
                rows[j]+=cols[i][j]
                if not i == len(headerRow)-1: rows[j]+=' '*columnSpacing

        line = '-'*len(header)
        print(line)
        print(header)
        print(line)
        for row in rows: print(row)
        print(line)

    def keep_pareto(pts):
        nd_pts = pts.sort_values(by='error_val')
        nd_pts = nd_pts.reset_index()

        for i in range(nd_pts.shape[0]):
            nd_pts.at[i,'pareto'] = 1

        for i in range(nd_pts.shape[0]):
           for j in range(nd_pts.shape[0]):
               if (i != j) & (nd_pts.at[j,'pareto'] == 1):
                   if (((nd_pts[i:i+1]['error_val'].values <  nd_pts[j:j+1]['error_val'].values) & (nd_pts[i:i+1][metric_to_plot].values <= nd_pts[j:j+1][metric_to_plot].values)) or
                       ((nd_pts[i:i+1]['error_val'].values <= nd_pts[j:j+1]['error_val'].values) & (nd_pts[i:i+1][metric_to_plot].values <  nd_pts[j:j+1][metric_to_plot].values))):
                       nd_pts.at[j,'pareto'] = 0

        return nd_pts
    #------------------------------------------------------------------------------

    paleta_colores_dt = ('orange', 'darkviolet', 'darkolivegreen', 'blue', 'black', 'turquoise')
    paleta_colores_lg = ('saddlebrown', 'maroon', 'green', 'darkolivegreen', 'brown', 'darkgreen')
    contador_dt = 0
    contador_lg = 0
    offset_color = 0

    for problem in problems:



        if problem == '':
            offset_color = offset_color + 1
            continue

        if problem in ['dataset_fdr_dt_bb', 'dataset_fdr_dt_cb']:
            metric_to_plot = 'false_discovery_rate_difference_value'
            metric_to_plot_test = 'false_discovery_rate_difference_value_tst'

            if name_experiment in ['E3-SP-T1T2', 'E3-EO-T1T2', 'E3-FDR-T1T2', 'E3-SP-T3T4']:
                output_type = 'binary' if ('_bb' in problem) or ('_rb' in problem) else 'continuous'
                line_name = 'PP_' + output_type
            else:
                line_name = 'PP'  # Predictive Parity
        elif problem in ['dataset_sp_dt_bb', 'dataset_sp_dt_cb']:
            metric_to_plot = 'statistical_parity_difference_value'
            metric_to_plot_test = 'statistical_parity_difference_value_tst'
            if name_experiment in ['E3-SP-T1T2', 'E3-EO-T1T2', 'E3-FDR-T1T2', 'E3-SP-T3T4']:
                output_type = 'binary' if ('_bb' in problem) or ('_rb' in problem) else 'continuous'
                if output_type == 'continuous':
                    line_name = 'RT'
                else:
                    line_name = 'CT'
                #line_name = r'$SP_C$' + "_" + output_type
            else:
                line_name = 'SP' # Statistical Parity
        elif problem in ['dataset_eo_dt_bb', 'dataset_eo_dt_cb']:
            metric_to_plot = 'equal_opportunity_difference_value'
            metric_to_plot_test = 'equal_opportunity_difference_value_tst'
            if name_experiment in ['E3-SP-T1T2', 'E3-EO-T1T2', 'E3-FDR-T1T2', 'E3-SP-T3T4']:
                output_type = 'binary' if ('_bb' in problem) or ('_rb' in problem)  else 'continuous'
                line_name = 'EO_' + output_type
            else:
                line_name = 'EO' # Equal Opportunity
        elif problem in ['dataset_sp_dt_rb', 'dataset_sp_dt_rc']:
            metric_to_plot = 'Average_Outcome_value'
            metric_to_plot_test = 'Average_Outcome_value_tst'
            if name_experiment in ['E3-SP-T1T2', 'E3-EO-T1T2', 'E3-FDR-T1T2', 'E3-SP-T3T4']:
                output_type = 'binary' if ('_bb' in problem) or ('_rb' in problem)  else 'continuous'
                if output_type == 'continuous':
                    line_name = 'RT'
                else:
                    line_name = 'CT'
                # line_name = r'$SP_R$' + "_" + output_type
            else:
                line_name = 'AO' # Average Outcome

    #    if ('german' in problem) and (contador_lg == 0):
    #        contador_lg = 1


        if not combine_plots:
            contador_dt = 0
            contador_lg = 0
            offset_color = 0

        ds = pd.read_csv(dsname[problem])

        plt.style.use('classic') #bmh  seaborn-bright

        error =[]
        dem_fp =[]
        subsets = []
        size = 0

        column = 'seed'

        #for i in range(1,11):
        for i in range(10):
            seed = 100+i
            print(seed, metric_to_plot)
        #    sb = ds.loc[(ds['generation']==200) & (ds[column]==seed) & (ds['rank']==0)]
            sb = ds.loc[(ds[column]==seed)]
            sb = sb.drop_duplicates(subset=['error_val', metric_to_plot])

        #    sb =  keep_pareto(sb)
        #    sb = sb.loc[(sb['pareto']==1)]
        #    sb.drop(columns=['pareto'])

            subsets.append(sb)
            size += subsets[i-1].shape[0]

            #size = max(size, subsets[i-1].shape[0])

        # el tamaño debe ser mínimo 3 para que de valores coherentes. Si no, da error.
        size = max (3, round(size/10))

        quantiles_plt = [i/(size-1) for i in range(0,size)]
        print('quantiles plt', quantiles_plt)
        quantiles_tab = [0, 0.25, 0.5, 0.75, 1]

        print(subsets)



    ##############    plt.axis(axisbelow=True)

        if 'dt' in problem:
            metrics_tab = ('error_val',metric_to_plot,'error_tst',metric_to_plot_test,'actual_depth','actual_leaves')
        else:
            metrics_tab = ('error_val',metric_to_plot,'error_tst',metric_to_plot_test,'max_iter','tol')

        metrics_plt = (metric_to_plot,'error_val')

        values_tab = {}
        values_plt = {}

        for i in range(10):
            seed = 100+i
            #seed = i
        #    subset = ds.loc[(ds['generation']==200) & (ds[column]==seed) & (ds['rank']==0)]
            subset = ds.loc[(ds[column]==seed) ] # & (ds['error_val']<1)
            # TODO por que tenemos error menor que 1 aqui hm
            subset = subset.drop_duplicates(subset=['error_val',metric_to_plot])


        #    subset =  keep_pareto(subset)
        #    subset = subset.loc[(subset['pareto']==1)]
        #    subset.drop(columns=['pareto'])

            subset = subset.sort_values(by='error_val')
            subset = subset.select_dtypes(include='number')


            if (len(subset)>0):
                for m in metrics_plt:
                    if (i>0):
                        if (m in (metric_to_plot,metric_to_plot_test)):
                            values_plt[m] = np.append(values_plt[m],[subset.quantile(quantiles_plt)[[m]].to_numpy().ravel()[::-1]],axis=0)
                        else:
                            values_plt[m] = np.append(values_plt[m],[subset.quantile(quantiles_plt)[[m]].to_numpy().ravel()],axis=0)
                    else:
                        if (m in (metric_to_plot,metric_to_plot_test)):
                            values_plt[m] = [subset.quantile(quantiles_plt)[[m]].to_numpy().ravel()[::-1]]
                        else:
                            values_plt[m] = [subset.quantile(quantiles_plt)[[m]].to_numpy().ravel()]

                for m in metrics_tab:
                    if (i>0):
                        if (m in (metric_to_plot,metric_to_plot_test)):
                            values_tab[m] = np.append(values_tab[m],[subset.quantile(quantiles_tab)[[m]].to_numpy().ravel()[::-1]],axis=0)
                        else:
                            values_tab[m] = np.append(values_tab[m],[subset.quantile(quantiles_tab)[[m]].to_numpy().ravel()],axis=0)
                    else:
                        if (m in (metric_to_plot,metric_to_plot_test)):
                            values_tab[m] = [subset.quantile(quantiles_tab)[[m]].to_numpy().ravel()[::-1]]
                        else:
                            values_tab[m] = [subset.quantile(quantiles_tab)[[m]].to_numpy().ravel()]

                if not combine_plots:
                    if (seed == 100):
                        plt.plot(subset[metric_to_plot], subset['error_val'], 'o', color='gray', alpha=0.25, markeredgewidth=0, zorder=10) #orange alpha=0.5
                    else:
                        plt.plot(subset[metric_to_plot], subset['error_val'], 'o', color='gray', alpha=0.25, markeredgewidth=0, zorder=10) #orange alpha=0.5

                plt.xlabel('FPR$_\mathrm{diff}$')
                # plt.xlabel('Fairness Measure')
                plt.xlabel('$SP_C$')
                plt.ylabel(ylabel)

        error = values_plt['error_val']
        error_tab = values_tab['error_val']
        dem_fp = values_plt[metric_to_plot]
        dem_fp_tab = values_tab[metric_to_plot]

        error_mean = np.mean(error,axis=0)
        error_stde = np.std(error,axis=0)

        error_iqr = np.percentile(error,75,axis=0) - np.percentile(error,25,axis=0)

        dem_fp_mean = np.mean(dem_fp,axis=0)
        dem_fp_stde = np.std(dem_fp,axis=0)

        means_tab = {}
        for m in metrics_tab:
            means_tab[m] = np.mean(values_tab[m],axis=0)

        if ('propublica' in problem):
            #ProPublica
            compas_val = np.array([
            [ 0.14670222332475427 , 0.3594213287164407 ],
            [ 0.10831275720164607 , 0.3746046279091937 ],
            [ 0.18534698343232262 , 0.34742730955147205 ],
            [ 0.11008904374758033 , 0.34378027446441817 ],
            [ 0.09107686281599325 , 0.3531929999247516 ],
            [ 0.10236355675369385 , 0.3314683352902924 ],
            [ 0.12243957355818993 , 0.3583608490085999 ],
            [ 0.12364454221131743 , 0.3329714080143794 ],
            [ 0.12622779844647497 , 0.34126761795067884 ],
            [ 0.13564778429357352 , 0.3576676604633988 ]
            ])
            compas_tst = np.array([
            [ 0.1322004942694598 , 0.34426418580962737 ],
            [ 0.16005907960199006 , 0.3291154154738961 ],
            [ 0.14695286322525383 , 0.3579866627361987 ],
            [ 0.16333209208595276 , 0.34547045365758167 ],
            [ 0.1169154228855721 , 0.3403063317239685 ],
            [ 0.11497545008183307 , 0.352922741089996 ],
            [ 0.18311456814130608 , 0.3597424863575409 ],
            [ 0.14066776135741654 , 0.340513184547367 ],
            [ 0.13144563489804761 , 0.35947491223239236 ],
            [ 0.1854262013729977 , 0.3461355593553611 ]
            ])
        elif ('violent' in problem):
            #Violent
            compas_val = np.array([
            [ 0.12629399585921322 , 0.30494712978683725 ],
            [ 0.1586738819431486 , 0.37314419180408087 ],
            [ 0.12664431673052365 , 0.3041855429204424 ],
            [ 0.1689309614285234 , 0.29396674918117405 ],
            [ 0.07606716886377901 , 0.3502190045664133 ],
            [ 0.1570049813200498 , 0.3227988706604582 ],
            [ 0.0793978188715031 , 0.3326435590498412 ],
            [ 0.16454737894334517 , 0.31703883029434665 ],
            [ 0.16859699114501947 , 0.3146916947630697 ],
            [ 0.12121020867026944 , 0.3252094132700495 ]
            ])
            compas_tst = np.array([
            [ 0.13195440777937195 , 0.3527977602278347 ],
            [ 0.11074493194802904 , 0.31731053272947773 ],
            [ 0.13623959785409537 , 0.35033051161637374 ],
            [ 0.13117185844458573 , 0.31185539776146076 ],
            [ 0.1355263157894737 , 0.3386014900951926 ],
            [ 0.15930138309247058 , 0.33618183340494734 ],
            [ 0.1272119693172325 , 0.3548595025034246 ],
            [ 0.22931880734958227 , 0.3223937953849625 ],
            [ 0.06859881609623764 , 0.33937693387519274 ],
            [ 0.15958193979933108 , 0.3256880341435816 ]
            ])

        if ('propublica' in problem) or ('violent' in problem):
            compas_fpr_v, compas_error_v = np.mean(compas_val[:,0]) , np.mean(compas_val[:,1])
            compas_fpr_t, compas_error_t = np.mean(compas_tst[:,0]) , np.mean(compas_tst[:,1])
            means_tab['error_val'] = np.append(means_tab['error_val'],compas_error_v)
            means_tab[metric_to_plot] = np.append(means_tab[metric_to_plot],compas_fpr_v)
            means_tab['error_tst'] = np.append(means_tab['error_tst'],compas_error_t)
            means_tab[metric_to_plot_test] = np.append(means_tab[metric_to_plot_test],compas_fpr_t)
            for m in metrics_tab:
                if not (m in ('error_val',metric_to_plot,'error_tst',metric_to_plot_test)):
                    means_tab[m] = np.append(means_tab[m],0)
            quant = ['  min','  Q1 (25%)','  Q2 (50%)','  Q3 (75%)','  max', 'COMPAS']
        else:
            quant = ['  min','  Q1 (25%)','  Q2 (50%)','  Q3 (75%)','  max']

        header = [problem,'Error_v','Unfairness_v','Error_t','Unfairness_t','Depth','Leaves']
        makeTable(header,[quant,means_tab[metrics_tab[0]],means_tab[metrics_tab[1]],means_tab[metrics_tab[2]],means_tab[metrics_tab[3]],means_tab[metrics_tab[4]],means_tab[metrics_tab[5]]])


        from scipy.interpolate import interp1d

        print(error_mean)
        print(dem_fp_mean)

        f = interp1d(dem_fp_mean, error_mean, kind='quadratic')
        x_new = np.linspace(dem_fp_mean.min(), dem_fp_mean.max(),500)
        y_smooth=f(x_new)

        zip2d = lambda a, b: [list(c) for c in zip(a, b)]
        pts2d = []
        for i in range(len(quantiles_plt)):
            pts2d.append(zip2d(error[:,i],dem_fp[:,i]))

        pts2d_mean = zip2d(error_mean,dem_fp_mean)

        import math

        dist_eucl = lambda a, b: math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]))

        dist = []
        for i in range(len(pts2d)):
            dist.append(0)
            for j in range(len(pts2d[i])):
                dist[i] += dist_eucl(pts2d_mean[i], pts2d[i][j])
            dist[i] /= len(pts2d[i])

        f = interp1d(dem_fp_mean, error_iqr, kind='quadratic')
        y_smooth_iqr=f(x_new)

        if 'dt' in problem:
            contador_dt = contador_dt + 1
            color_pareto = paleta_colores_dt[contador_dt-1+offset_color]
            label_line = line_name  # 'decision tree' problem + '_' +
        else:
            contador_lg = contador_lg + 1
            label_line = 'logistic regression'
            color_pareto = paleta_colores_lg[contador_lg-1+offset_color]

            if (prefix is not None):
                label_line = label_line + prefix[contador_lg-1]
            else:
                if contador_lg > 1:
                    label_line = label_line + ' ' + str(contador_lg)

        plt.plot(x_new, y_smooth, '-', color=color_pareto, alpha=0.6, markeredgewidth=0, zorder=11, label=label_line)
        plt.plot(dem_fp_mean, error_mean, 'o', color=color_pareto, alpha=0.6, markeredgewidth=0, zorder=11)

        # Printing pareto measures, X -> fairness measure Y -> error
        if pareto_metrics:
            df_pareto = pd.DataFrame({
                'X': dem_fp_mean,
                'Y': error_mean
            })

            pareto_q_measures = print_pareto_measures(df_pareto)
            pareto_q_measures['dataset'] = dataset
            pareto_q_measures['outputs'] = dsname[problem].split('_')[4]
            pareto_q_measures['objectives_names'] = dsname[problem].split('_')[1]
            pareto_q_measures['measures'] = metric_to_plot

            pareto_metrics_dataframe = pd.concat([pareto_metrics_dataframe, pareto_q_measures])

        if plot_shadow:
            #plt.fill_between(dem_fp_mean, error_mean-error_iqr,error_mean+error_iqr,linewidth=0,color='blue',alpha=.1)
            plt.fill_between(x_new, y_smooth-y_smooth_iqr,y_smooth+y_smooth_iqr,linewidth=0,color=color_pareto,alpha=.2) #'blue'

        if True: #combine_plots:
            leg = plt.legend()
            leg_lines = leg.get_lines()
            leg_texts = leg.get_texts()
            plt.setp(leg_lines, linewidth=3, markeredgewidth=1)
            plt.setp(leg_texts, fontsize='small')

        '''
        res = 200
        for i in range(1,res):
            plt.fill_between(x_new, y_smooth+y_smooth_iqr*(i-1)/res,y_smooth+y_smooth_iqr*i/res,linewidth=0,color='blue',alpha=.6*(res-i)/res)
            plt.fill_between(x_new, y_smooth-y_smooth_iqr*(i-1)/res,y_smooth-y_smooth_iqr*i/res,linewidth=0,color='blue',alpha=.6*(res-i)/res)
        '''

        #plt.errorbar(dem_fp_mean, error_mean, yerr=error_iqr)

        from matplotlib.ticker import StrMethodFormatter
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

        if not combine_plots:
            if ('dt' in problem):
                if ('adult' in problem):
                    plt.xlim(left=-0.003,right=0.10)
                    plt.ylim(bottom=0.15,top=0.51)
                elif ('german' in problem):
                    plt.xlim(left=-0.01)
                    plt.ylim(bottom=0.185,top=0.58)
                elif ('propublica' in problem):
                    plt.xlim(left=-0.005)
                    plt.ylim(bottom=0.3,top=0.425)
                elif ('violent' in problem):
                    plt.xlim(left=-0.005)
                    plt.ylim(bottom=0.29,top=0.53)
                elif ('ricci' in problem):
                    plt.xlim(right=1.05)
            else: #lg
                if ('adult' in problem):
                    plt.xlim(left=-0.005,right=0.176)
                    plt.ylim(bottom=0.19,top=0.66)
                elif ('german' in problem):
                    plt.xlim(left=-0.01,right=0.36)
                    plt.ylim(bottom=0.19,top=0.67)
                elif ('propublica' in problem):
                    plt.xlim(left=-0.005,right=0.16)
                    plt.ylim(bottom=0.29,top=0.81)
                elif ('violent' in problem):
                    plt.xlim(left=-0.005,right=0.18)
                    plt.ylim(bottom=0.26,top=0.88)
                elif ('ricci' in problem):
                    plt.xlim(right=1.05)
                    plt.ylim(top=0.35)
        else: #combine_plots
            if ('dt' in problem):
                if ('adult' in problem):
                    #plt.xlim(left=-0.003,right=0.07)
                    #plt.ylim(top=0.56)
                    plt.xlim(left=-0.004,right=0.15)
                    plt.ylim(bottom=0.16,top=0.57)
                elif ('german' in problem):
                    plt.xlim(left=-0.01)
                    plt.ylim(bottom=0.24,top=0.61)
                elif ('propublica' in problem):
                    #plt.xlim(left=-0.005)
                    #plt.ylim(bottom=0.315,top=0.8)
                    plt.xlim(left=0,right=0.13)
                    plt.ylim(bottom=0.29,top=0.81)
                elif ('violent' in problem):
                    plt.xlim(left=-0.005)
                    plt.ylim(bottom=0.29,top=0.71)
                elif ('ricci' in problem):
                    plt.xlim(right=1.05)
            else: #lg
                if ('adult' in problem):
                    plt.xlim(left=-0.004,right=0.15)
                    #plt.ylim(bottom=0.16,top=0.57)
                    plt.ylim(bottom=0.16,top=0.67) #f1score
                elif ('german' in problem):
                    #plt.xlim(left=-0.006,right=0.27)
                    #plt.ylim(bottom=0.235,top=0.605)
                    plt.xlim(left=-0.0,right=0.27) #f1score
                    plt.ylim(bottom=0.235,top=0.616) #f1score
                elif ('propublica' in problem):
                    #plt.xlim(left=0,right=0.13)
                    plt.xlim(left=-0.005,right=0.13) #f1score
                    plt.ylim(bottom=0.29,top=0.81)
                elif ('violent' in problem):
                    plt.xlim(left=-0.004,right=0.14)
                    #plt.ylim(bottom=0.285,top=0.72)
                    plt.ylim(bottom=0.285,top=0.85) #f1score
                elif ('ricci' in problem):
                    plt.xlim(right=1.05)


        #subset = ds.loc[(ds['generation']==200) & (ds['rank']==0)]
        #subset =  keep_pareto(subset)
        #pareto = subset.loc[(subset['pareto']==1)]
        #plt.plot(pareto['dem_fp'], pareto['error'], '-o', color='blue', alpha=0.5, markeredgecolor="white", markeredgewidth=0.5)
        #plt.plot(pareto['dem_fp'], pareto['error'], '-o', color='orange', alpha=0.5, markeredgewidth=0) #color='orange',
        #plt.grid(color='grey', linestyle=':', linewidth=0.2, solid_joinstyle='round')

        plt.minorticks_on()
        plt.grid(color='grey', which='minor', linestyle=':', linewidth=0.15, alpha=0.5, zorder=0) #, solid_joinstyle='round'
        plt.grid(color='grey', which='major', linestyle='-', linewidth=0.25, alpha=0.5, zorder=0)

        '''
        from sklearn.svm import SVR
        svr_rbf = SVR(kernel='rbf', C=1e4, gamma=0.1)
        svr_lin = SVR(kernel='linear', C=1e4)
        svr_poly = SVR(kernel='poly', C=1e4, degree=4)
        
        X=dem_fp.ravel().reshape(-1,1)
        y=error.ravel()
        
        y_rbf = svr_rbf.fit(X,y).predict(X)
        #y_lin = svr_lin.fit(X,y).predict(X)
        #y_poly = svr_poly.fit(X,y).predict(X)
        
        plt.plot(dem_fp.ravel(),y_rbf,'o',color='red')
        '''

        #plt.axes(axisbelow=True)

        if ('propublica' in problem) or ('violent' in problem):
            plt.plot(np.mean(compas_val[:,0]), np.mean(compas_val[:,1]), 'o', color='red', alpha=1, markeredgewidth=0, zorder=11)
            plt.hlines(np.mean(compas_val[:,1]),plt.xlim()[0],np.mean(compas_val[:,0]),linestyles='dotted',color='red')
            plt.vlines(np.mean(compas_val[:,0]),plt.ylim()[0],np.mean(compas_val[:,1]),linestyles='dotted',color='red')

        bbox = dict(boxstyle="round", fc="0.8")
        arrowprops = dict(
            arrowstyle = "->",
            connectionstyle = "angle,angleA=0,angleB=90,rad=3")

        #plt.annotate('COMPAS:\n(%.2f, %.2f)'%(np.mean(_compas[:,0]), np.mean(_compas[:,1])), (np.mean(_compas[:,0]),np.mean(_compas[:,1])*1.005),  xytext=(1*offset, offset), textcoords='offset points',bbox=bbox, arrowprops=arrowprops)
        offset = 50
        if combine_plots:
            if ('propublica' in problem) or ('violent' in problem):
                plt.annotate('COMPAS', (compas_fpr_v,compas_error_v+0.0005), xytext=(-1.5*offset, offset), textcoords='offset points',bbox=bbox, arrowprops=arrowprops, zorder=15)
        else:
            if ('violent' in problem):
                plt.annotate('COMPAS', (compas_fpr_v,compas_error_v+0.0005), xytext=(0.2*offset, offset), textcoords='offset points',bbox=bbox, arrowprops=arrowprops, zorder=15) #violent
            elif ('propublica' in problem):
                plt.annotate('COMPAS', (compas_fpr_v,compas_error_v+0.0005), xytext=(0.6*offset, offset), textcoords='offset points',bbox=bbox, arrowprops=arrowprops, zorder=15) #propublica

        plt.title(titles[problem])
        plt.title('LSAC')

        plt.savefig(name_experiment + '_' + dataset+'_'+metric_to_plot+'.pdf')
        #plt.savefig(problem+'.png')
        #plt.show()

        if not combine_plots:
            plt.close()

        #----------------------------------

        if print_params:
            #sb = ds.drop_duplicates(subset=['error_val',metric_to_plot])
            sb = ds.drop_duplicates(subset=['error_val',metric_to_plot,'seed'])

            size = 100
            quantiles_plt = [i/(size-1) for i in range(0,size)]

            ######## Histogramas
            #param = ['actual_depth','actual_leaves','min_samples_split','class_weight','criterion']
            param = ['class_weight']

            from matplotlib.ticker import FormatStrFormatter

            ax1_col = 'error_tst'
            ax2_col = metric_to_plot_test

            sb['class_weight'] = 10 - sb['class_weight']

            for p in param:
                means = sb.groupby([p],as_index=False).mean()
                variance = sb.groupby([p],as_index=False).std()

                fig,ax1 = plt.subplots()

                width = 0.4

                #ax.bar(means[p]-width*0.5,means['error_tst'],width,color='blue',yerr=variance['error_tst'],ecolor='plum',zorder=10)
                ax1.errorbar(means[p],means[ax1_col],yerr=variance[ax1_col],color='plum',zorder=10)
                ax1.scatter(means[p],means[ax1_col],color='blue',zorder=11)

                ax2=ax1.twinx()
                #ax2.bar(means[p]+width*0.5,means[metric_to_plot_test],width,color='red',yerr=variance[metric_to_plot_test],ecolor='salmon',zorder=10)
                #ax2.plot(means[p],means[ax2_col],'red')
                ax2.errorbar(means[p],means[ax2_col],yerr=variance[ax2_col],color='salmon',zorder=10)
                ax2.scatter(means[p],means[ax2_col],color='red',zorder=11)

                ax1.set_xlabel(p)

                ax1.set_ylabel('1$-$G-mean (test)',color='blue')
                ax1.tick_params(axis='y', colors='blue')
                ax2.set_ylabel('FPR$_\mathrm{diff}$ (test)',color='red') #,fontsize=14
                ax2.tick_params(axis='y', colors='red')
                ax2.set_xlim(means[p].min()-width,means[p].max()+width)

                #ax.grid(color='grey', which='major', linestyle='', axis='x')
                ax1.grid(color='grey', which='major', linestyle=':', axis='y', linewidth=0.25, alpha=0.5, zorder=0)
                #ax2.grid(color='red', which='major', linestyle=':', axis='y', linewidth=0.25, alpha=0.5, zorder=0)

                #ax.set_zorder(ax2.get_zorder()+1)
                #ax.patch.set_visible(False)

                ax1.autoscale(enable=True,tight=True,axis='y')
                ax2.autoscale(tight=True)

                #ax2_min = means[metric_to_plot_test].min()
                ax1_min = ax1.get_ylim()[0]
                ax1_max = means[ax1_col].max()+variance[ax1_col].max()
                ax1_max += (ax1_max-ax1_min)*0.05
                ax1_stp = (ax1_max-ax1_min)/6
                ax1.set_yticks(np.arange(ax1_min,ax1_max*1.01,step=ax1_stp))
                ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                #ax2_min = means[metric_to_plot_test].min()
                ax2_min = ax2.get_ylim()[0]
                ax2_max = means[metric_to_plot_test].max()+variance[ax2_col].max()
                ax2_max += (ax2_max-ax2_min)*0.05
                ax2_stp = (ax2_max-ax2_min)/6
                ax2.set_yticks(np.arange(ax2_min,ax2_max*1.01,step=ax2_stp))
                ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                if p in ('class_weight'):
                    #ax.set_ylim(top=0.815)
                    #ax2.set_ylim(top=0.105)
                    ax1.set_xlabel('Class weight')
                    ax1.set_xlim(left=2.5,right=9.5)
                    #ax2.set_ylim(top=ax2_max)
                    #ax.set_ylim(top=ax_max)

                plt.savefig(problem+'_'+p+'.pdf')

            if 'dt' in problem:
                #param = ['actual_depth','actual_leaves','min_samples_split','class_weight','criterion']
                #param = ['actual_depth','actual_leaves','min_samples_split']
                param = ['min_samples_split']
            else:
                param = ['C']

            ax1_col = 'error_tst'
            ax2_col = metric_to_plot_test

            for p in param:
                x_smooth = np.linspace(sb[p].min(),sb[p].max(),50)

                for i in range(10):
                #for i in range(0,1,1):
                    seed = 100+i
                    ss = sb.loc[(sb[column]==seed)]
                    f1 = interp1d(ss[p], ss[ax1_col], kind='linear',bounds_error=False,fill_value=(3,3))
                    f2 = interp1d(ss[p], ss[ax2_col], kind='linear',bounds_error=False,fill_value=(3,3))
                    xx = x_smooth[(x_smooth >= ss[p].min()) & (x_smooth <= ss[p].max())]
                    y1 = f1(xx)
                    y2 = f2(xx)

                    extremes = ss.groupby([p],as_index=False).mean()
                    y1[0] = extremes[ax1_col][0]
                    y2[0] = extremes[ax2_col][0]

                    xy12 = np.column_stack((xx,y1,y2))
                    if (i==0):
                        data = np.vstack([xy12])
                    else:
                        data = np.vstack([data,xy12])

                df = pd.DataFrame({p:data[:,0],ax1_col:data[:,1],ax2_col:data[:,2]})

                xy_mean=df.groupby([p],as_index=False).mean()
                xy_stde=df.groupby([p],as_index=False).std()

                x_plt     = xy_mean[p]
                y1_mn_plt = xy_mean[ax1_col]
                y1_sd_plt = xy_stde[ax1_col]
                np.nan_to_num(y1_sd_plt,copy=False)
                y2_mn_plt = xy_mean[ax2_col]
                y2_sd_plt = xy_stde[ax2_col]
                np.nan_to_num(y2_sd_plt,copy=False)

                fig,ax = plt.subplots()

                x_smooth2 = np.linspace(x_plt.min(), x_plt.max(),500)

                f1_m = interp1d(x_plt, y1_mn_plt, kind='quadratic')
                y1_m_smooth = f1_m(x_smooth2)

                f1_s = interp1d(x_plt, y1_sd_plt, kind='quadratic') #slinear quadratic cubic
                y1_sd_smooth = f1_s(x_smooth2)

                f2_m = interp1d(x_plt, y2_mn_plt, kind='quadratic')
                y2_m_smooth = f2_m(x_smooth2)

                f2_s = interp1d(x_plt, y2_sd_plt, kind='quadratic')
                y2_sd_smooth = f2_s(x_smooth2)

                #y2_m_smooth=np.clip(y2_m_smooth,0,y2_m_smooth.max())

                ax.plot(x_smooth2,y1_m_smooth,color='blue')
                ax.fill_between(x_smooth2,y1_m_smooth-y1_sd_smooth,y1_m_smooth+y1_sd_smooth,linewidth=0,color='blue',alpha=.1)
                ax.set_xlabel(p)
                ax.set_ylabel('1$-$G-mean (test)',color='blue')
                ax.yaxis.label.set_color('blue')
                ax.tick_params(axis='y', colors='blue')

                ax2=ax.twinx()
                ax2.plot(x_smooth2,y2_m_smooth,color='red')
                #ax2.plot(x_plt,y2_mn_plt,color='red')
                ax2.fill_between(x_smooth2,y2_m_smooth-y2_sd_smooth,y2_m_smooth+y2_sd_smooth,linewidth=0,color='red',alpha=.1)
                ax2.set_ylabel('FPR$_\mathrm{diff}$ (test)',color='red') #,fontsize=14
                ax2.tick_params(axis='y', colors='red')
                ax2.set_xlim(right=x_plt.max())

                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                if p in ('actual_depth'):
                    ax.set_xlabel('Depth')
                    ax2.set_ylim(bottom=-.01,top=0.175)
                elif p in ('actual_leaves'):
                    ax.set_xlabel('Leaves')
                    if (problem in ('propublica')):
                        ax.set_xlim(right=600)
                        ax.set_ylim(top=0.6)
                    elif (problem in ('violent')):
                        ax.set_xlim(left=11)
                        ax2.set_ylim(bottom=0.017,top=0.10)
                elif p in ('min_samples_split'):
                    ax.set_xlabel('Min samples split')
                    if (problem in ('violent')):
                        ax.set_xlim(left=6)
                        ax2.set_ylim(bottom=0)
                elif p in ('C'):
                    ax.set_xlabel('C')
                    if (problem in ('violent')):
                        ax.set_xlim(left=6)
                        ax2.set_ylim(bottom=0)

                #ax.minorticks_on()
                #ax.grid(color='grey', which='minor', linestyle=':', linewidth=0.15, alpha=0.5, zorder=0) #, solid_joinstyle='round'

                #ax.plot(x_plt,y1_mn_plt,'o', color='blue', alpha=0.5, markeredgewidth=0, markersize=3, zorder=11)

                ax.grid(color='grey', which='major', linestyle='-', axis='x', linewidth=0.25, alpha=0.5, zorder=0)
                ax.grid(color='grey', which='major', linestyle=':', axis='y', linewidth=0.25, alpha=0.5, zorder=0)
                #ax2.grid(color='grey', which='major', linestyle='--', linewidth=0.25, alpha=0.5, zorder=0)

                plt.savefig(problem+'_'+p+'.pdf')
        #end if params-----------------------------------

    #end for problems------------------------------------

if pareto_metrics:
    pareto_metrics_dataframe.reset_index(drop=True, inplace=True)
    pareto_metrics_dataframe.to_csv('pareto_metrics_dataframe_T2_{}.csv'.format(dataset), index=False)