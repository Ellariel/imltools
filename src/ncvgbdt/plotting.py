'''
Plotting of the results of a fitted :class:`ncvgbdt.NCVGBDTRegressor` /
:class:`ncvgbdt.NCVGBDTClassifier`.

The single public entry point is :func:`make_figures`. It takes a fitted
estimator or the directory of a stored one and writes figures (PNG,
optionally SVG).

See the package README for design choices and caveats relevant for
interpreting the figures (shuffled-target baseline, corrected t-test,
SHAP effects, ...).
'''

import os
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t
from shap import Explanation
from shap import dependence_plot
from shap.plots import beeswarm
from shap.plots import scatter
from sklearn.metrics import confusion_matrix

from .fitting import create_dir


def _total_compute_time(explainations):
    '''Sum of the SHAP compute times (missing times count as 0).'''
    return float(np.sum([k.compute_time or 0 for k in explainations]))


def format_p(p, add_p=True, keep_space=False):
    if not np.isfinite(p) or p < 0.0:
        p = 'p = inf'
    elif p >= 0.999:
        p = 'p = 1.000'
    elif p < 0.001:
        p = 'p < .001'
    else:
        p = 'p = ' + f"{p:.3f}"[1:]
    p = p if add_p else p.replace('p ', '').replace('=', '')
    p = p if keep_space else p.replace(' ', '')
    return p

def format_r(r):
    if not np.isfinite(r):
        r = 'inf'
    elif abs(r) >= 0.999:
        r = '1.00'
    elif abs(r) < 0.005:
        r = '0.00'
    else:
        r = f"{'-' if r < 0.0 else ''}"+f"{abs(r):.2f}"[1:]
    return r

def get_stars(p, p001='***', p01='** ', p05='*  ', p10='⁺  ', p_='    '):
    if p < 0.001:
        return p001
    if p < 0.010:
        return p01
    if p < 0.050:
        return p05
    if p < 0.100:
        return p10
    return p_


def corrected_std(differences, n_tst_over_n_trn=0.25):
    '''
    Corrects standard deviation using Nadeau and Bengio's approach.
    Ref: Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/
    plot_grid_search_stats.html

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_tst_over_n_trn : float
        Number of samples in the testing set over number of samples in the
        training set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    '''

    # Get corrected std -------------------------------------------------------
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    # Corrected variance
    corrected_var = np.var(differences, ddof=1) * (1/kr + n_tst_over_n_trn)
    # Corrected standard deviation
    corrected_std = np.sqrt(corrected_var)

    # Return corrected standard deviation -------------------------------------
    return corrected_std


def corrected_ttest(differences, n_tst_over_n_trn=0.25):
    '''
    Computes right-tailed paired t-test with corrected variance (H1: mean
    of ``differences`` > 0).
    Ref: Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/
    plot_grid_search_stats.html

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_tst_over_n_trn : float
        Number of samples in the testing set over number of samples in the
        training set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    '''

    # Compute t statistics and p value ----------------------------------------
    # Get mean of differences
    mean = np.mean(differences)
    # Get corrected standard deviation
    std = corrected_std(differences, n_tst_over_n_trn)
    # Compute t statistics
    if std >= 0.0 and std <= 0.0000001: # for technical purposes
        t_stat = 0.0
        p_val = 1.0
    else:
        # Compute p value for right-tailed t-test (H1: mean > 0). A negative
        # t statistic (first model worse than second) is not significant.
        t_stat = mean / std
        p_val = t.sf(t_stat, df=len(differences)-1)
    # Return t statistics and p value -----------------------------------------
    return t_stat, p_val


def print_parameter_distributions(task, results, plots_path):
    '''
    Print model parameter distributions in histogram.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # Get params
    params = pd.DataFrame(results['best_params'])

    # Make plot ---------------------------------------------------------------
    # Iterate over columns of params dataframe
    for (name, data) in params.items():
        # Make figure
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot hist of inlier score
        sns.histplot(data=data.astype('float'),
                     bins=30,
                     kde=True,
                     color='#777777',
                     ax=ax)
        # Remove top, right and left frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add x label
        ax.set_xlabel(name)
        # Add y label
        ax.set_ylabel('Number')
        # Set title
        ax.set_title(task['ANALYSIS_NAME']+' ' +
                     'parameter distribution for predicting'+' ' +
                     task['y_name'][0],
                     fontsize=10)

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     '0'+'_' +
                     task['y_name'][0]+'_' +
                     'hyperparameter'+'_' +
                     name)[:130]
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        #plt.show()
        plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_regression_scatter(task, results, plots_path):
    '''
    Print model fit in a scatter plot (regression).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # True values
    true_values = np.concatenate([i['y_true'] for i in results['scores']])
    # Predicted values
    pred_values = np.concatenate([i['y_pred'] for i in results['scores']])

    # Make plot ---------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Print data
    ax.scatter(pred_values,
               true_values,
               zorder=2,
               alpha=0.1,
               color='#444444')
    # Add optimal fit line
    ax.plot([-10000, 10000], [-10000, 10000],
            color='#999999',
            zorder=3,
            linewidth=2,
            alpha=0.3)
    # Fix aspect
    ax.set_aspect(1)
    # Remove top, right and left frame elements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Remove ticks
    ax.tick_params(axis='both', which='major', reset=True,
                   bottom=True, top=False, left=True, right=False)
    # Add grid
    ax.grid(visible=True, which='major', axis='both')
    # Modify grid
    ax.tick_params(grid_linestyle=':', grid_alpha=.5)
    # Get true values range
    true_values_range = max(true_values) - min(true_values)
    # Set x-axis limits
    ax.set_xlim(min(true_values) - true_values_range/20,
                max(true_values) + true_values_range/20)
    # Set y-axis limits
    ax.set_ylim(min(true_values) - true_values_range/20,
                max(true_values) + true_values_range/20)
    # Set title
    ax.set_title(task['ANALYSIS_NAME']+' ' +
                 'predicting'+' ' +
                 task['y_name'][0],
                 fontsize=10)
    # Set xlabel
    ax.set_xlabel('Predicted values', fontsize=10)
    # Set x ticks size
    plt.xticks(fontsize=10)
    # Set ylabel
    ax.set_ylabel('True values', fontsize=10)
    # Set y ticks size
    plt.yticks(fontsize=10)

    # Add MAE -----------------------------------------------------------------
    # Extract MAE
    mae = [i['mae'] for i in results['scores']]
    # Extract MAE shuffle
    mae_sh = [i['mae'] for i in results['scores_sh']]
    # Calculate p-value between MAE and shuffle MAE
    _, pval_mae = corrected_ttest(np.array(mae_sh)-np.array(mae))
    # Add MAE results to plot
    ax.text(.40, .055, ('MAE original mean'+r'$\pm$'+'std:{:.2f}'+r'$\pm$' +
            '{:.2f}|med:{:.2f}').format(
            np.mean(mae),
            np.std(mae),
            np.median(mae)),
            transform=ax.transAxes,
            fontsize=8)
    # Add MAE p val results to plot
    ax.text(.40, .02, ('MAE shuffle mean'+r'$\pm$'+'std:{:.2f}'+r'$\pm$' +
            '{:.2f}|med:{:.2f}|{}').format(
            np.mean(mae_sh),
            np.std(mae_sh),
            np.median(mae_sh),
            format_p(pval_mae)),
            transform=ax.transAxes,
            fontsize=8)

    # Add R² ------------------------------------------------------------------
    # Extract R²
    r2 = [i['r2'] for i in results['scores']]
    # Extract R² shuffle
    r2_sh = [i['r2'] for i in results['scores_sh']]
    # Calculate p-value between R² and shuffle R²
    _, pval_r2 = corrected_ttest(np.array(r2)-np.array(r2_sh))
    # Add R² results to plot
    ax.text(.02, .96, ('R² original mean'+r'$\pm$'+'std:{:.3f}'+r'$\pm$' +
            '{:.3f}|med:{:.3f}').format(
            np.mean(r2),
            np.std(r2),
            np.median(r2)),
            transform=ax.transAxes,
            fontsize=8)
    # Add R² p val results to plot
    ax.text(.02, .925, ('R² shuffle mean'+r'$\pm$'+'std:{:.3f}'+r'$\pm$' +
            '{:.3f}|med:{:.3f}|{}').format(
            np.mean(r2_sh),
            np.std(r2_sh),
            np.median(r2_sh),
            format_p(pval_r2)),
            transform=ax.transAxes,
            fontsize=8)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 '1'+'_' +
                 task['y_name'][0])[:130]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg',  bbox_inches='tight')
    # Show figure
    #plt.show()
    plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_regression_violin(task, results, plots_path):
    '''
    Print model fit in a violin plot (regression).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # Extract MAE
    mae = [i['mae'] for i in results['scores']]
    # Extract MAE shuffle
    mae_sh = [i['mae'] for i in results['scores_sh']]
    # Extract R²
    r2 = [i['r2'] for i in results['scores']]
    # Extract R² shuffle
    r2_sh = [i['r2'] for i in results['scores_sh']]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {'Mean Absolute Error': pd.Series(np.array(mae)),
         'R2': pd.Series(np.array(r2)),
         'Data': pd.Series(['original' for _ in mae]),
         'Dummy': pd.Series(np.ones(np.array(mae).shape).flatten())})
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {'Mean Absolute Error': pd.Series(np.array(mae_sh)),
         'R2': pd.Series(np.array(r2_sh)),
         'Data': pd.Series(['shuffle' for _ in mae_sh]),
         'Dummy': pd.Series(np.ones(np.array(mae_sh).shape).flatten())})
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ['Mean Absolute Error', 'R2']

    # Make plot ---------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, len(metrics)*.75+1))
    # Set tight figure layout
    fig.tight_layout()
    # Make color palette
    mypal = {'original': '#777777', 'shuffle': '#eeeeee'}
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        sns.violinplot(x=metric, y='Dummy', hue='Data', data=all_scores_df,
                       bw_method='scott', cut=2, density_norm='width', gridsize=100,
                       width=0.8, inner='box', orient='h', linewidth=1,
                       saturation=1, ax=ax[i], palette=mypal)
        # Remove top, right and left frame elements
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        # Remove ticks
        ax[i].tick_params(axis='both', which='major', reset=True,
                          bottom=True, top=False, left=False, right=False,
                          labelleft=False)
        # Set x ticks and size
        ax[i].set_xlabel(metrics[i], fontsize=10)
        # Set y ticks and size
        ax[i].set_ylabel('', fontsize=10)
        # For other than first metric
        if i > 0:
            # Remove legend
            ax[i].legend().remove()
        # Add horizontal grid
        fig.axes[i].set_axisbelow(True)
        # Set grid style
        fig.axes[i].grid(axis='y', color='#bbbbbb', linestyle='dotted',
                         alpha=.3)
    # Make title string
    title_str = (
        task['ANALYSIS_NAME']+' ' +
        'predicting'+' ' +
        task['y_name'][0])
    # set title
    fig.axes[0].set_title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 '1'+'_' +
                 task['y_name'][0]+'_' +
                 'distribution')[:130]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show plot
    #plt.show()
    plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_classification_confusion(task, results, plots_path):
    '''
    Print model fit as confusion matrix (classification).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # True values
    true_values = [i['y_true'] for i in results['scores']]
    # Predicted values
    pred_values = [i['y_pred'] for i in results['scores']]
    # Sample weights  list
    sample_weights = [i['class_weights'] for i in results['scores']]
    # Accuracy
    acc = [i['acc'] for i in results['scores']]
    # Schuffle accuracy
    acc_sh = [i['acc'] for i in results['scores_sh']]
    # Get classes
    class_labels = np.unique(np.concatenate(true_values)).tolist()

    # Get confusion matrix ----------------------------------------------------
    # Loop over single results
    for true, pred, w in zip(true_values, pred_values, sample_weights):
        if 'con_mat' not in locals():
            # Compute confusion matrix
            con_mat = confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=np.array([w[i] for i in true]),
                normalize='all')
        else:
            # Add confusion matrix
            con_mat = np.add(con_mat, confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=np.array([w[i] for i in true]),
                normalize='all'))
    # Normalize confusion matrix
    con_mat_norm = con_mat / len(true_values)

    # Plot confusion matrix ---------------------------------------------------
    # Create figure
    fig, ax = plt.subplots(figsize=(con_mat.shape[0]*.5+3,
                                    con_mat.shape[0]*.5+3))
    # Plot confusion matrix
    sns.heatmap(con_mat_norm*100,
                vmin=None,
                vmax=None,
                cmap='Greys',
                center=None,
                robust=True,
                annot=True,
                fmt='.2f',
                annot_kws={'size': 10},
                linewidths=1,
                linecolor='#999999',
                cbar=False,
                cbar_kws=None,
                square=True,
                xticklabels=class_labels,
                yticklabels=class_labels,
                mask=None,
                ax=ax)
    # Add x label to plot
    plt.xlabel('Predicted class', fontsize=10)
    # Add y label to plot
    plt.ylabel('True class', fontsize=10)
    # Set y ticks size and sets the yticks 'upright' with 0
    plt.yticks(rotation=0, fontsize=10)
    # Calculate p-value of accuracy and shuffle accuracy
    _, pval_acc = corrected_ttest(np.array(acc)-np.array(acc_sh))
    # Make title string
    title_str = (
        task['ANALYSIS_NAME']+' ' +
        'predicting'+' ' +
        task['y_name'][0]+'\n' +
        'Orig. data accuracy mean'+r'$\pm$'+'std|median: {:.2f}'+r'$\pm$' +
        '{:.2f}|{:.2f}'+'\n' +
        'Shuffle data accuracy mean'+r'$\pm$'+'std|median: {:.2f}'+r'$\pm$' +
        '{:.2f}|{:.2f}'+'\n' +
        'p-value of orig. and shuffle: {}'+'\n').format(
        np.mean(acc)*100,
        np.std(acc)*100,
        np.median(acc)*100,
        np.mean(acc_sh)*100,
        np.std(acc_sh)*100,
        np.median(acc_sh)*100,
        format_p(pval_acc))
    # Set title
    plt.title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 '1'+'_' +
                 task['y_name'][0])[:130]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show figure
    #plt.show()
    plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_classification_violin(task, results, plots_path):
    '''
    Print model fit in a violin plot (classification).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # Extract accuracy
    acc = [i['acc'] for i in results['scores']]
    # Extract shuffle accuracy
    acc_sh = [i['acc'] for i in results['scores_sh']]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {'Accuracy': pd.Series(np.array(acc)),
         'Data': pd.Series(['original' for _ in acc]),
         'Dummy': pd.Series(np.ones(np.array(acc).shape).flatten())})
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {'Accuracy': pd.Series(np.array(acc_sh)),
         'Data': pd.Series(['shuffle' for _ in acc_sh]),
         'Dummy': pd.Series(np.ones(np.array(acc_sh).shape).flatten())})
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ['Accuracy']

    # Make plot ---------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(figsize=(8, len(metrics)*.75+1))
    # Make color palette
    mypal = {'original': '#777777', 'shuffle': '#eeeeee'}
    # Put ax into list
    ax = [ax]
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        sns.violinplot(x=metric, y='Dummy', hue='Data', data=all_scores_df,
                       bw_method='scott', cut=2, density_norm='width', gridsize=100,
                       width=0.8, inner='box', orient='h', linewidth=1,
                       saturation=1, ax=ax[i], palette=mypal)
        # Remove top, right and left frame elements
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        # Remove ticks
        ax[i].tick_params(axis='both', which='major', reset=True,
                          bottom=True, top=False, left=False, right=False,
                          labelleft=False)
        # Set x ticks and size
        ax[i].set_xlabel(metrics[i], fontsize=10)
        # Set y ticks and size
        ax[i].set_ylabel('', fontsize=10)
        # For other than first metric
        if i > 0:
            # Remove legend
            ax[i].legend().remove()
        # Add horizontal grid
        fig.axes[i].set_axisbelow(True)
        # Set grid style
        fig.axes[i].grid(axis='y', color='#bbbbbb', linestyle='dotted',
                         alpha=.3)
    # Make title string
    title_str = (
        task['ANALYSIS_NAME']+' ' +
        'predicting'+' ' +
        task['y_name'][0])
    # set title
    plt.title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 '1'+'_' +
                 task['y_name'][0]+'_' +
                 'distribution')[:130]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show plot
    #plt.show()
    plt.close()

    # Return None -------------------------------------------------------------
    return fig


def get_shap_effects(task, explainations, c_class=-1):
    '''
    Get SHAP based global effects.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    explainations : list of shap explaination objects
        SHAP explaination holding the results of the ml analyses.
    c_class : integer
        Current class for slicing.

    Returns
    -------
    shap_effects : list
        SHAP effects.
    shap_base : float
        Base value corresponds to expected value of the predictor.
    '''

    # Get shap effects --------------------------------------------------------
    # Case 1: no interaction and regression
    if not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # SHAP effects
        shap_effects = [np.mean(np.abs(k.values), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 2: interaction and regression
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # Get SHAP effects
        shap_effects = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 3: no interaction and binary
    elif not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # SHAP effects
        shap_effects = [np.mean(np.abs(k.values[:, :, c_class]), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values
                                  for k in explainations]))
    # Case 4: interaction and binary
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # Get SHAP effects
        shap_effects = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 5: no interaction and multiclass
    elif not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # SHAP effects
        shap_effects = [np.mean(np.abs(k.values[:, :, c_class]), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values
                                  for k in explainations]))
    # Case 6: interaction and multiclass
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # Get SHAP effects
        shap_effects = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case else
    else:
        # Raise error
        raise ValueError('Unsupported task.')

    # Make SHAP effects dataframe
    shap_effects_df = pd.DataFrame(shap_effects, columns=task['x_names'])

    # Return shap effects -----------------------------------------------------
    return shap_effects_df, base


def print_shap_effects(task, results, plots_path):
    '''
    Print SHAP based global effects.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap effects -------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(task,
                                                 results['explainations'],
                                                 c_class)
        # Get current shuffle shap effects
        shap_effects_sh_df, _ = get_shap_effects(task,
                                                 results['explainations_sh'],
                                                 c_class)

        # Process SHAP effects-------------------------------------------------
        # Mean shap values
        shap_effects_se_mean = shap_effects_df.mean(axis=0)
        # Sort from highto low
        shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(
            ascending=True)

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])

        # Plot ----------------------------------------------------------------
        # Make horizontal bar plot
        shap_effects_se_mean_sort.plot(
            kind='barh',
            figsize=(x_names_max_len*.1+7, x_names_count*.4+1),
            color='#777777',
            fontsize=10)
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel('mean(|SHAP values|)', fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Remove top, right and left frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add horizontal grid
        ax.set_axisbelow(True)
        # Set grid style
        ax.grid(axis='y', color='#bbbbbb', linestyle='dotted', alpha=.3)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'SHAP effects for'+' ' +
            task['y_name'][0]+'\n' +
            'mean(|SHAP values|) = mean absolute deviation from expected' +
            ' value (' +
            str(np.round(base, decimals=2)) +
            ')'
            )
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            title_str = title_str+' class: '+str(c_class)
        # Set title
        ax.set_title(title_str, fontsize=10)

        # Compute SHAP effect p values ----------------------------------------
        # Init p value list
        pval = []
        # Iterate over predictors
        for pred_name, pred_data in shap_effects_df.items():
            # Get current p value
            _, c_pval = corrected_ttest(
                pred_data.to_numpy()-shap_effects_sh_df[pred_name].to_numpy())
            # Add to pval list
            pval.append(np.around(c_pval, decimals=3))
        # Make pval series
        pval_se = pd.Series(data=pval, index=task['x_names'])
        # Multiple comparison correction
        if task['MCC']:
            # Multiply p value by number of tests
            pval_se = pval_se*x_names_count
            # Set p values > 1 to 1
            pval_se = pval_se.clip(upper=1)

        # Add SHAP effect values and p values as text -------------------------
        # Loop over values
        for i, (c_pred, c_val) in enumerate(shap_effects_se_mean_sort.items()):
            # Make test string
            txt_str = (str(np.around(c_val, decimals=2))+'|' +
                       #'p '+str(pval_se[c_pred]))
                       format_p(pval_se[c_pred]))
            # Add values to plot
            ax.text(c_val, i, txt_str, color='k',
                    va='center', fontsize=8)
        # Get x limits
        x_left, x_right = plt.xlim()
        plt.xlim(x_left, x_right + x_right*.1)

        # Save plot -----------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     '2'+'_' +
                     task['y_name'][0]+'_' +
                     'shap_effects')[:130]
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            save_path = save_path+'_class_'+str(c_class)
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        #plt.show()
        plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_shap_effects_distribution(task, results, plots_path):
    '''
    Print SHAP values distribution.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap effects distribution ------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(task,
                                                 results['explainations'],
                                                 c_class)
        # Get current shuffle shap effects
        shap_effects_sh_df, _ = get_shap_effects(task,
                                                 results['explainations_sh'],
                                                 c_class)

        # Process SHAP effects-------------------------------------------------
        # Sorting index by mean value of columns
        i_srt = shap_effects_df.mean().sort_values(ascending=False).index
        # Sort SHAP effects dataframe
        shap_effects_df_sort = shap_effects_df.reindex(i_srt, axis=1)
        # Sort shuffle SHAP effects dataframe
        shap_effects_sh_df_sort = shap_effects_sh_df.reindex(i_srt, axis=1)
        # Add data origin to SHAP effects dataframe
        shap_effects_df_sort['Data'] = pd.DataFrame(
            ['original' for _ in range(shap_effects_df_sort.shape[0])],
            columns=['Data'])
        # Add data origin to shuffle SHAP effects dataframe
        shap_effects_sh_df_sort['Data'] = pd.DataFrame(
            ['shuffle' for _ in range(shap_effects_sh_df_sort.shape[0])],
            columns=['Data'])
        # Get value name
        value_name = 'mean(|SHAP value|)'
        # Melt SHAP effects dataframe
        shap_effects_df_sort_melt = shap_effects_df_sort.melt(
            id_vars=['Data'], var_name='predictors',
            value_name=value_name)
        # Melt shuffle SHAP effects dataframe
        shap_effects_sh_df_sort_melt = shap_effects_sh_df_sort.melt(
            id_vars=['Data'], var_name='predictors',
            value_name=value_name)
        # Concatenate importances dataframes
        shap_effects_df_sort_melt_all = pd.concat([
            shap_effects_df_sort_melt,
            shap_effects_sh_df_sort_melt], axis=0)

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])

        # Plot ----------------------------------------------------------------
        # Make figure
        fig, ax = plt.subplots(figsize=(x_names_max_len*.1+7,
                                        x_names_count*.4+1))
        # Make color palette
        mypal = {'original': '#777777', 'shuffle': '#eeeeee'}
        # Plot data
        sns.violinplot(x=value_name, y='predictors', hue='Data',
                       data=shap_effects_df_sort_melt_all, bw_method='scott',
                       cut=2, density_norm='width', gridsize=100, width=0.8,
                       inner='box', orient='h', linewidth=.5,
                       saturation=1, ax=ax, palette=mypal)
        # Get the current figure and axes objects.
        _, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel('mean(|SHAP values|)', fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel('', fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Remove top, right and left frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add horizontal grid
        ax.set_axisbelow(True)
        # Set grid style
        ax.grid(axis='y', color='#bbbbbb', linestyle='dotted', alpha=.3)
        # Set legend position
        plt.legend(loc='lower right')
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'SHAP effects distribution for'+' ' +
            task['y_name'][0]+'\n' +
            'mean(|SHAP values|) = mean absolute deviation from expected' +
            ' value (' +
            str(np.round(base, decimals=2)) +
            ')'
            )
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            title_str = title_str+' class: '+str(c_class)
        # Add title
        ax.set_title(title_str, fontsize=10)

        # Save plots and results ----------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     '2'+'_' +
                     task['y_name'][0]+'_' +
                     'shap_effects_distribution')[:130]
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            save_path = save_path+'_class_'+str(c_class)
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg', bbox_inches='tight')
        # Show figure
        # plt.show()
        plt.close()

    # Return None -------------------------------------------------------------
    return fig


def get_shap_values(task, explainations, c_class=-1):
    '''
    Get SHAP values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    explainations : list of shap explaination objects
        SHAP explaination holding the results of the ml analyses.
    c_class : integer
        Current class for slicing.

    Returns
    -------
    shap_explainations : shap explaination object
        Explaination object with SHAP values.
    shap_base : float
        Base value corresponds to expected value of the predictor.
    '''

    # Get shap effects --------------------------------------------------------
    # Case 1: no interaction and regression
    if not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k.values for k in explainations]),
            base_values=np.hstack([k.base_values for k in explainations]),
            data=np.vstack([k.data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=_total_compute_time(explainations))
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 2: interaction and regression
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k.values for k in explainations]),
            base_values=np.hstack([k.base_values for k in explainations]),
            data=np.vstack([k.data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=_total_compute_time(explainations))
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 3: no interaction and binary
    elif not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k[:, :, c_class].values for k in explainations]),
            base_values=np.hstack([k[:, :, c_class].base_values
                                   for k in explainations]),
            data=np.vstack([k[:, :, c_class].data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=_total_compute_time(explainations))
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values
                                  for k in explainations]))
    # Case 4: interaction and binary
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k.values for k in explainations]),
            base_values=np.hstack([k.base_values for k in explainations]),
            data=np.vstack([k.data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=_total_compute_time(explainations))
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 5: no interaction and multiclass
    elif not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k[:, :, c_class].values for k in explainations]),
            base_values=np.hstack([k[:, :, c_class].base_values
                                   for k in explainations]),
            data=np.vstack([k[:, :, c_class].data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=_total_compute_time(explainations))
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values
                                  for k in explainations]))
    # Case 6: interaction and multiclass
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k.values for k in explainations]),
            base_values=np.hstack([k.base_values for k in explainations]),
            data=np.vstack([k.data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=_total_compute_time(explainations))
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case else
    else:
        # Raise error
        raise ValueError('Unsupported task.')

    # Return shap effects -----------------------------------------------------
    return shap_explainations, base


def print_shap_values(task, results, plots_path):
    '''
    Plot SHAP values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task, results['explainations'],
                                            c_class)

        # If interactions
        if task['SHAP_INTERACTIONS']:
            # Sum over interaction to get full effects
            shap_explainations = shap_values.sum(axis=2)
            # Add base values
            shap_explainations.base_values = shap_values.base_values
            # Add data
            shap_explainations.data = shap_values.data
        # Other
        else:
            shap_explainations = shap_values

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])

        # Plot SHAP values beeswarm -------------------------------------------
        beeswarm(shap_explainations,
                 max_display=len(task['x_names']),
                 order=Explanation.abs.mean(0),
                 clustering=None,
                 cluster_threshold=0.5,
                 color=None,
                 axis_color='#333333',
                 alpha=.66,
                 show=False,
                 log_scale=False,
                 color_bar=True,
                 plot_size=(x_names_max_len*.1+7, x_names_count*.4+1),
                 color_bar_label='Predictor value')
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel('SHAP values', fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'SHAP values for'+' ' +
            task['y_name'][0]+'\n' +
            'mean(|SHAP values|) = mean absolute deviation from expected' +
            ' value (' +
            str(np.round(base, decimals=2)) +
            ')'
            )
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            title_str = title_str+' class: '+str(c_class)
        # Add title
        plt.title(title_str, fontsize=10)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel('Predictor value', fontsize=10)

        # Save plot -----------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     '3'+'_' +
                     task['y_name'][0]+'_' +
                     'shap_values')[:130]
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            save_path = save_path+'_class_'+str(c_class)
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        # plt.show()
        plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_shap_dependences(task, results, plots_path):
    '''
    Plot SHAP dependences.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task, results['explainations'],
                                            c_class)
        # If interactions
        if task['SHAP_INTERACTIONS']:
            # Sum over interaction to get full effects
            shap_explainations = shap_values.sum(axis=2)
            # Add base values
            shap_explainations.base_values = shap_values.base_values
            # Add data
            shap_explainations.data = shap_values.data
        # Other
        else:
            shap_explainations = shap_values

        # Print shap values dependencies --------------------------------------
        # Loop over predictors
        for i, c_pred in enumerate(shap_explainations.feature_names):
            # Make figure
            fig, ax = plt.subplots(figsize=(8, 5))
            # Plot SHAP Scatter plot
            scatter(shap_explainations[:, i],
                    color='#777777',
                    hist=True,
                    axis_color='#333333',
                    dot_size=16,
                    x_jitter='auto',
                    alpha=.5,
                    title=None,
                    xmin=None,
                    xmax=None,
                    ymin=None,
                    ymax=None,
                    overlay=None,
                    ax=ax,
                    ylabel='SHAP values',
                    show=False)
            # Get the current figure and axes objects.
            _, ax = plt.gcf(), plt.gca()
            # Set x label size
            plt.xlabel(ax.get_xlabel(), fontsize=10)
            # Set x ticks size
            plt.xticks(fontsize=10)
            # Set y label size
            plt.ylabel(ax.get_ylabel(), fontsize=10)
            # Set y ticks size
            plt.yticks(fontsize=10)
            # Make title string
            title_str = (
                task['ANALYSIS_NAME']+' ' +
                'SHAP values for'+' ' +
                task['y_name'][0]+'\n' +
                'SHAP values = deviation from expected value (' +
                str(np.round(base, decimals=2)) +
                ')'
                )
            # Add class if no interactions and binary or multiclass
            if not task['SHAP_INTERACTIONS'] and (
                    task['OBJECTIVE'] == 'binary' or
                    task['OBJECTIVE'] == 'multiclass'):
                # Make title string
                title_str = title_str+' class: '+str(c_class)
            # Add title
            plt.title(title_str, fontsize=10)

            # Save plot -------------------------------------------------------
            # Make save path
            save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                         '4'+'_' +
                         task['y_name'][0]+'_' +
                         'shap_values_dependency'+'_' +
                         str(c_pred))[:130]
            # Add class if no interactions and binary or multiclass
            if not task['SHAP_INTERACTIONS'] and (
                    task['OBJECTIVE'] == 'binary' or
                    task['OBJECTIVE'] == 'multiclass'):
                # Make title string
                save_path = save_path+'_class_'+str(c_class)
            # Save figure
            plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
            # Check if save as svg is enabled
            if task['AS_SVG']:
                # Save figure
                plt.savefig(save_path+'.svg',  bbox_inches='tight')
            # Show figure
            # plt.show()
            plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_shap_effects_interactions(task, results, plots_path):
    '''
    Plot SHAP effects inclusive interactions.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap effects -------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(task,
                                                 results['explainations'],
                                                 c_class)

        # Process SHAP effects-------------------------------------------------
        # Mean shap values
        shap_effects_se_mean = shap_effects_df.mean(axis=0)
        # Sort from highto low
        shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(
            ascending=False)

        # Get SHAP effects interactions ---------------------------------------
        # SHAP effects
        shap_effects_inter = np.array([np.mean(np.abs(k.values), axis=0)
                                       for k in results['explainations']])
        # Make dataframe
        shap_effects_inter_df = pd.DataFrame(
            np.mean(shap_effects_inter, axis=0),
            index=task['x_names'],
            columns=task['x_names'])
        # Reindex to sorted index
        shap_effects_inter_sort_df = \
            shap_effects_inter_df.reindex(shap_effects_se_mean_sort.index)
        # Reorder columns to sorted index
        shap_effects_inter_sort_df = \
            shap_effects_inter_sort_df.loc[:, shap_effects_se_mean_sort.index]
        # SHAP effects shuffle
        shap_effects_inter_sh = np.array(
            [np.mean(np.abs(k.values), axis=0) for k in
             results['explainations_sh']])

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])

        # Make labels with pvales ---------------------------------------------
        # Get p values
        pval = np.zeros((shap_effects_inter.shape[1],
                         shap_effects_inter.shape[2]))
        # Iterate over shap_effects
        for x, y in np.ndindex((shap_effects_inter.shape[1],
                                shap_effects_inter.shape[2])):
            # Get current SHAP effect
            c_effect = shap_effects_inter[:, x, y]
            # Get current SHAP effect shuffle
            c_effect_sh = shap_effects_inter_sh[:, x, y]
            # Calculate p-value
            _, pval[x, y] = corrected_ttest(c_effect-c_effect_sh)
        # Multiple comparison correction
        if task['MCC']:
            # Multiply p value by number of tests
            pval = pval*(x_names_count**2)
            # Set p values > 1 to 1
            pval = pval.clip(None, 1)
        # Initialize labels dataframe
        interaction_labels_df = pd.DataFrame(np.zeros([
            shap_effects_inter.shape[1],
            shap_effects_inter.shape[2]]), dtype=object)
        # Iterate labels
        for x, y in np.ndindex((shap_effects_inter.shape[1],
                                shap_effects_inter.shape[2])):
            # Make label
            interaction_labels_df.iloc[x, y] = (
                str(np.around(shap_effects_inter_df.iloc[x, y],
                              decimals=2)) +
                '\n'+'p'+' ' +
                str(np.around(pval[x, y], decimals=3)))
        # Index labels dataframe
        interaction_labels_df.index = shap_effects_inter_df.index
        # Column labels
        interaction_labels_df.columns = shap_effects_inter_df.columns
        # Reindex to sorted index
        interaction_labels_sort_df = \
            interaction_labels_df.reindex(shap_effects_se_mean_sort.index)
        # Reorder columns to sorted index
        interaction_labels_sort_df = \
            interaction_labels_sort_df.loc[:, shap_effects_se_mean_sort.index]

        # Plot interaction effects --------------------------------------------
        # Create figure
        fig, ax = plt.subplots(figsize=(x_names_max_len*.1+x_names_count*1+1,
                                        x_names_max_len*.1+x_names_count*1+1))
        # Make colorbar string
        clb_str = ('mean(|SHAP value|)')
        # Plot confusion matrix
        sns.heatmap(shap_effects_inter_sort_df,
                    vmin=None,
                    vmax=None,
                    cmap='Greys',
                    center=None,
                    robust=True,
                    annot=interaction_labels_sort_df,
                    fmt='',
                    annot_kws={'size': 10},
                    linewidths=1,
                    linecolor='#999999',
                    cbar=True,
                    cbar_kws={'label': clb_str, 'shrink': 0.6},
                    square=True,
                    xticklabels=True,
                    yticklabels=True,
                    mask=None,
                    ax=ax)
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel(ax.get_xlabel(), fontsize=10)
        # Set x ticks size
        plt.xticks(rotation=90, fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(rotation=0, fontsize=10)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'SHAP effects for'+' ' +
            task['y_name'][0]+'\n' +
            'mean(|SHAP values|) = deviation from expected value (' +
            str(np.round(np.mean(np.hstack(
                [k.base_values for k in results['explainations']])),
                decimals=2)) +
            ')'
            )
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            title_str = title_str+' class: '+str(c_class)
        # Add title
        plt.title(title_str, fontsize=10)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel(clb_str, fontsize=10)
        cb_ax.set_box_aspect(50)

        # Save plot -----------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     '5'+'_' +
                     task['y_name'][0]+'_' +
                     'shap_effects_interactions')[:130]
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            save_path = save_path+'_class_'+str(c_class)
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        #plt.show()
        plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_shap_interaction_values(task, results, plots_path):
    '''
    Plot SHAP interaction values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task, results['explainations'],
                                            c_class)

        # Print shap values dependencies --------------------------------------
        # Make list of permutations
        permutations_list = (
            [(i, i) for i in shap_values.feature_names] +
            list(permutations(shap_values.feature_names, 2)))
        # Loop over predictor pairs
        for ind in permutations_list:
            # Make figure
            fig, ax = plt.subplots(figsize=(8, 5))
            # Plot SHAP dependence
            dependence_plot(ind,
                            shap_values=shap_values.values,
                            features=pd.DataFrame(
                                shap_values.data,
                                columns=shap_values.feature_names),
                            feature_names=shap_values.feature_names,
                            display_features=None,
                            interaction_index='auto',
                            color='#1E88E5',
                            axis_color='#333333',
                            cmap=None,
                            dot_size=16,
                            x_jitter=0,
                            alpha=.66,
                            title=None,
                            xmin=None,
                            xmax=None,
                            ax=ax,
                            show=False,
                            ymin=None,
                            ymax=None)
            # Get the current figure and axes objects.
            _, ax = plt.gcf(), plt.gca()
            # Set x label size
            ax.set_xlabel(ax.get_xlabel(), fontsize=10)
            # Set x ticks size
            plt.xticks(fontsize=10)
            # Set y label size
            plt.ylabel(ax.get_ylabel(), fontsize=10)
            # Set y ticks size
            plt.yticks(fontsize=10)
            # Make title string
            title_str = (
                task['ANALYSIS_NAME']+' ' +
                'SHAP interaction values for'+' ' +
                task['y_name'][0]+'\n' +
                'SHAP values = deviation from expected value (' +
                str(np.round(np.mean(np.hstack(
                    [k.base_values for k in results['explainations']])),
                    decimals=2)) +
                ')'
                )
            # Add class if no interactions and binary or multiclass
            if not task['SHAP_INTERACTIONS'] and (
                    task['OBJECTIVE'] == 'binary' or
                    task['OBJECTIVE'] == 'multiclass'):
                # Make title string
                title_str = title_str+' class: '+str(c_class)
            # Add title
            ax.set_title(title_str, fontsize=10)
            # Check if mor than 1 axes are present
            if len(fig.axes) > 1:
                # Get colorbar
                cb_ax = fig.axes[1]
                # Modifying color bar tick size
                cb_ax.tick_params(labelsize=10)
                # Modifying color bar fontsize
                cb_ax.set_ylabel(cb_ax.get_ylabel(), fontsize=10)

            # Save plot -------------------------------------------------------
            # Make save path
            save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                         '6'+'_' +
                         task['y_name'][0]+'_' +
                         'shap_interaction_values'+'_' +
                         ind[0]+'_' +
                         ind[1])[:130]
            # Add class if no interactions and binary or multiclass
            if not task['SHAP_INTERACTIONS'] and (
                    task['OBJECTIVE'] == 'binary' or
                    task['OBJECTIVE'] == 'multiclass'):
                # Make title string
                save_path = save_path+'_class_'+str(c_class)
            # Save figure
            plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
            # Check if save as svg is enabled
            if task['AS_SVG']:
                # Save figure
                plt.savefig(save_path+'.svg',  bbox_inches='tight')
            # Show figure
            #plt.show()
            plt.close()

    # Return None -------------------------------------------------------------
    return fig

def ilm_shap_values_graph(task, results, plots_path='./', title=None, exclude=[], rename_dict={}, dpi=600, stars=False):
    '''
    Plot SHAP values.
    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.
    Returns
    -------
    None.
    '''
    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1
    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task, results['explainations'], c_class)
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(task,
                                                 results['explainations'],
                                                 c_class)
        # Get current shuffle shap effects
        shap_effects_sh_df, _ = get_shap_effects(task,
                                                 results['explainations_sh'],
                                                 c_class)
        # Process SHAP effects-------------------------------------------------
        # Mean shap values
        shap_effects_se_mean = shap_effects_df.mean(axis=0)
        # Sort from highto low
        shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(
            ascending=True)
        # If interactions
        if task['SHAP_INTERACTIONS']:
            # Sum over interaction to get full effects
            shap_explainations = shap_values.sum(axis=2)
            # Add base values
            shap_explainations.base_values = shap_values.base_values
            # Add data
            shap_explainations.data = shap_values.data
        # Other
        else:
            shap_explainations = shap_values
        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        # Compute SHAP effect p values ----------------------------------------
        # Init p value list
        pval = {}
        # Iterate over predictors
        for pred_name, pred_data in shap_effects_df.items():
            # Get current p value
            _, c_pval = corrected_ttest(
                pred_data.to_numpy()-shap_effects_sh_df[pred_name].to_numpy())
            # Add to pval list
            pval[pred_name] = np.around(c_pval, decimals=3)
        excl = 0
        for e in exclude:
            idx = shap_explainations.feature_names.index(e)
            shap_explainations.feature_names.remove(e)
            shap_explainations.values = np.delete(shap_explainations.values, idx, axis=1)
            shap_explainations.data = np.delete(shap_explainations.data, idx, axis=1)
            excl += 1
            print(f"'{e}' has been removed.")
        # Make pval series
        x_names_count = len(task['x_names'])
        pval_se = pd.Series(data=pval, index=task['x_names'])
        # Multiple comparison correction
        if task['MCC']:
            # Multiply p value by number of tests
            pval_se = pval_se*(x_names_count - excl)
            # Set p values > 1 to 1
            pval_se = pval_se.clip(upper=1)
        extra_x_names = {}
        extra_x_names_max_len = []
        for i, (c_pred, c_val) in enumerate(shap_effects_se_mean_sort.items()):
            # Make test string
             txt_str = c_pred+'  '+f"{c_val:6<.2f}"+f"{', '+format_p(pval_se[c_pred]) if not stars else get_stars(pval_se[c_pred])}"
             extra_x_names[c_pred] = txt_str
        for i, f in enumerate(shap_explainations.feature_names):
            shap_explainations.feature_names[i] = extra_x_names[f].replace(f, rename_dict[f]) if f in rename_dict else extra_x_names[f]
            extra_x_names_max_len.append(len(shap_explainations.feature_names[i].split('\n')[0]))
        extra_x_names_max_len = max(extra_x_names_max_len)
        
        # Plot SHAP values beeswarm -------------------------------------------
        beeswarm(shap_explainations,
                 max_display=len(task['x_names']),
                 order=Explanation.abs.mean(0),
                 clustering=None,
                 cluster_threshold=0.5,
                 color=None,
                 axis_color='#333333',
                 alpha=.66,
                 show=False,
                 log_scale=False,
                 color_bar=True,
                 plot_size=(x_names_max_len*.05+7, x_names_count*.4+1),
                 color_bar_label='Predictor value')
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel('Impact on model output\n(SHAP value of predictor)', fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'SHAP values for'+' ' +
            task['y_name'][0]+'\n' +
            'mean(|SHAP values|) = mean absolute deviation from expected' +
            ' value (' +
            str(np.round(base, decimals=2)) +
            ')'
            )
        # Add class if no interactions and binary or multiclass
        
        if not (task['OBJECTIVE'] == 'binary' or
                        task['OBJECTIVE'] == 'multiclass'):
            # Extract R²
            r2 = [i['r2'] for i in results['scores']]
              # Extract R² shuffle
            r2_sh = [i['r2'] for i in results['scores_sh']]
              # Calculate p-value between R² and shuffle R²
            _, pval_r2 = corrected_ttest(np.array(r2)-np.array(r2_sh))
                # Add R² results to plot
            metrics_str = f'R²pred={format_r(np.mean(r2))}'+r'±'+f'{format_p(np.std(r2), add_p=False)}, {format_p(pval_r2, keep_space=False)}'
        else:
            metrics_str = 'Acc.='

        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            title_str = title_str+' class: '+str(c_class)
        # Add title
        print(title_str)
        print(metrics_str)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel('Predictor value', fontsize=10)
        # Add SHAP effect values and p values as text -------------------------
        # Loop over values
        x_left, x_right = plt.xlim()
        y_bottom, y_top = plt.ylim()
        #for i, (c_pred, c_val) in enumerate(shap_effects_se_mean_sort.items()):
            # Make test string
        #    txt_str = (str(np.around(c_val, decimals=2))+', '+format_p(pval_se[c_pred])) #get_stars(pval_se[c_pred])
            # Add values to plot
        #    ax.text(x_left-.15, i-.05, txt_str, color='k', va='center', fontsize=9) #c_val
        # Get x limits
        s = 'Average impact   \nPredictor     (mean|SHAP values|)'
        ax.text(x_left  - x_left*0.05, y_top, s, color='k', va='center', ha='right', fontsize=10) # - (max([len(i) for i in s.split('\n')])*.02 / 2)
        plt.xlim(x_left - x_left*0.05, x_right + x_right*.1)

        if title!=None:
          plt.title(title, fontsize=10)
        else:
          f = task['y_name'][0]
          f = f.replace(f, rename_dict[f]) if f in rename_dict else f
          f += f'\n{metrics_str}'
          ax.text((x_left + x_right) / 2, y_top + len(f.split('\n'))*.4, f, fontsize=10.6, 
                  horizontalalignment='center', verticalalignment='top')

        # Save plot -----------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     '3'+'_' +
                     task['y_name'][0]+'_' +
                     'shap_values')[:130]
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            save_path = save_path+'_class_'+str(c_class)
        # Save figure
        plt.savefig(save_path+'.png', dpi=dpi, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg', dpi=dpi,  bbox_inches='tight')
        # Show figure
        # plt.show()
        plt.close()
    # Return None -------------------------------------------------------------
    return fig

def make_figures(model,
                 figures_dir=None,
                 print_param_distrib=False,
                 print_shap_effects_inter=True,
                 add_multiple_comparison_correction=False,
                 def_shap_values=True,
                 as_svg=False):
    '''
    Draw the figures of a fitted estimator.

    A sub-directory ``<figures_dir>/<target>_plots`` is created holding:

    * model fit: regression scatter (predicted vs. true) and violin plots of
      R2 / MAE against the shuffled-target baseline; or, for classification,
      a class-weighted confusion matrix and violin plot of the balanced
      accuracy against the baseline,
    * SHAP effects (mean absolute SHAP value per predictor) and their
      distribution over outer folds, with the baseline from the shuffled
      model,
    * SHAP value plots (beeswarm) and dependence plots,
    * if the estimator used ``shap_interactions=True``: pairwise interaction
      effects and interaction value plots.

    For binary and multiclass tasks effects are drawn per class (for binary
    tasks class 0 is the mirror image of class 1).

    The p values printed in the figures come from a one-sided corrected
    resampled t-test (Nadeau & Bengio, 2003) of the real model against the
    shuffled-target model over the outer folds, see the package README for
    caveats.

    Parameters
    ----------
    model : NCVGBDTRegressor, NCVGBDTClassifier, str or path
        A fitted estimator, or the directory of a stored one (see
        :func:`ncvgbdt.load_model`).
    figures_dir : str, path or None, default None
        Directory in which ``<target>_plots`` is created (created if
        missing). None: ``<output_dir>/figures`` of the estimator, or the
        current directory if the estimator has no ``output_dir``.
    print_param_distrib : bool, default False
        Also plot the distributions of the tuned hyperparameters over folds.
    print_shap_effects_inter : bool, default True
        Plot the interaction figures if the estimator computed SHAP
        interaction values. Set False to skip them (they are slow).
    add_multiple_comparison_correction : bool, default False
        Bonferroni correction: multiply every SHAP-effect p value by the
        number of predictors (for interaction effects by the number of
        predictors squared), capped at 1.
    def_shap_values : bool, default True
        True: standard SHAP beeswarm plot. False: alternative graph of SHAP
        values (:func:`ilm_shap_values_graph`).
    as_svg : bool, default False
        Additionally store every figure as SVG.

    Returns
    -------
    str
        Path of the directory holding the figures.

    Raises
    ------
    sklearn.exceptions.NotFittedError
        If the estimator is not fitted.
    '''

    # Get the fitted estimator ------------------------------------------------
    if isinstance(model, (str, os.PathLike)):
        from .estimator import load_model
        model = load_model(model)
    if not hasattr(model, 'results_'):
        from sklearn.exceptions import NotFittedError
        raise NotFittedError('Fit the estimator before plotting.')
    if figures_dir is None:
        out = getattr(model, 'output_dir', None)
        figures_dir = os.path.join(out, 'figures') if out else './'

    # Task and results --------------------------------------------------------
    task = dict(model.task_)
    # Add multiple comparison correction to task
    task['MCC'] = add_multiple_comparison_correction
    # Add as svg to task
    task['AS_SVG'] = as_svg
    results = model.results_
    # Plots path
    plots_path = os.path.join(figures_dir, task['y_name'][0]+'_plots')
    # Create plots dir
    create_dir(plots_path)

    # Plot parameter distributions --------------------------------------------
    if print_param_distrib:
        print_parameter_distributions(task, results, plots_path)

    # Plot model fit ----------------------------------------------------------
    # Regressor
    if task['OBJECTIVE'] == 'regression':
        # Print model fit as scatter plot
        print_regression_scatter(task, results, plots_path)
        # Print model fit as violinplot of metrics
        print_regression_violin(task, results, plots_path)
    # Classification
    elif task['OBJECTIVE'] in ('binary', 'multiclass'):
        # Print model fit as confusion matrix
        print_classification_confusion(task, results, plots_path)
        # Print model fit as violinplot of metrics
        print_classification_violin(task, results, plots_path)
    # Other
    else:
        # Raise error
        raise ValueError('OBJECTIVE not found.')

    # Plot SHAP effects -------------------------------------------------------
    print_shap_effects(task, results, plots_path)

    # Plot SHAP effects distribution ------------------------------------------
    print_shap_effects_distribution(task, results, plots_path)

    # Plot SHAP values --------------------------------------------------------
    if def_shap_values:
        print_shap_values(task, results, plots_path)
    else:
        ilm_shap_values_graph(task, results, plots_path)

    # Plot SHAP dependencies --------------------------------------------------
    print_shap_dependences(task, results, plots_path)

    # Plot SHAP effects interactions ------------------------------------------
    if task['SHAP_INTERACTIONS'] and print_shap_effects_inter:
        print_shap_effects_interactions(task, results, plots_path)

    # Plot SHAP interaction values --------------------------------------------
    if task['SHAP_INTERACTIONS'] and print_shap_effects_inter:
        print_shap_interaction_values(task, results, plots_path)

    # Return path of the figures ----------------------------------------------
    return plots_path
