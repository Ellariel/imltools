"""
Plotting of the results of a fitted :class:`ncvgbdt.NCVGBDTRegressor` /
:class:`ncvgbdt.NCVGBDTClassifier`.

:func:`make_figures` takes a fitted estimator or the directory of a stored one
and returns the figures as ``{name: (fig, ax)}``, so that they can be edited;
:func:`save_figures` writes them as PNG, SVG or PDF files (``make_figures``
does this itself if given an ``output_dir``).

See the package README for design choices and caveats relevant for
interpreting the figures (shuffled-target baseline, corrected t-test,
SHAP effects, ...).
"""

import hashlib
import os
import re
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shap import Explanation
from shap import dependence_plot
from shap.plots import beeswarm
from shap.plots import scatter
from sklearn.metrics import confusion_matrix

from .fitting import CORRECTIONS
from .fitting import adjust_pvalues
from .fitting import corrected_std  # noqa: F401
from .fitting import corrected_ttest
from .fitting import effect_pvalues
from .fitting import get_interaction_effects
from .fitting import get_shap_effects
from .fitting import get_shap_values

from .format import format_p, format_r, get_stars


PLOT_TYPES = (
    "fit",
    "shap_effects",
    "shap_effects_distribution",
    "shap_values",
    "dependence",
    "interaction_effects",
    "interaction_values",
    "parameters",
)
DEFAULT_PLOT_TYPES = ("fit", "shap_effects", "shap_effects_distribution", "shap_values")
INTERACTION_PLOT_TYPES = ("interaction_effects", "interaction_values")
# Figure kinds whose drawing call accepts keyword arguments from ``plot_kwargs``
PLOT_KWARGS_KINDS = (
    "fit_scatter",
    "fit_violin",
    "fit_confusion",
    "shap_effects",
    "shap_effects_distribution",
    "shap_values",
    "dependence",
    "interaction_effects",
    "interaction_values",
    "parameters",
)
FIG_FORMATS = ("png", "svg", "pdf")


class _Figures:
    """Collects the figures of one :func:`make_figures` call."""

    def __init__(self, figsize=None, plot_kwargs=None):
        self.figures = {}
        self.figsize = figsize
        self.plot_kwargs = plot_kwargs or {}

    def kwargs(self, kind):
        """Keyword arguments the user gave for a figure kind (a copy)."""
        return dict(self.plot_kwargs.get(kind, {}))

    def add(self, name, plot_type, fig, ax):
        """Register a finished figure under ``name``."""
        if name in self.figures:
            raise ValueError(f"Duplicate figure name {name!r}.")
        size = self.figsize
        if isinstance(size, dict):
            size = size.get(plot_type)
        if size is not None:
            fig.set_size_inches(*size)
        self.figures[name] = (fig, ax)


def _call(fn, user, *args, **defaults):
    """Call ``fn``; keyword arguments in ``user`` override ``defaults``."""
    return fn(*args, **{**defaults, **user})


def _shap_call(fn, user, *args, **defaults):
    """
    Like :func:`_call` for SHAP plots, which draw into the *current* axes.
    Start from a fresh figure so that earlier (still open) figures are not
    drawn over.
    """
    fresh = plt.figure()
    result = _call(fn, user, *args, **defaults)
    if plt.gcf() is not fresh:  # the function made its own figure
        plt.close(fresh)
    return result


def _name(task, base, c_class=None):
    """
    Figure name: ``base`` plus ``:class_<label>`` for per-class figures.
    """
    name = base + task.get("NAME_SUFFIX", "")
    if (
        c_class is not None
        and not task["SHAP_INTERACTIONS"]
        and task["OBJECTIVE"] in ("binary", "multiclass")
    ):
        labels = task.get("CLASS_LABELS")
        name += f":class_{c_class if labels is None else labels[c_class]}"
    return name


def _class_loop(task, n_classes):
    """Class indices to draw (all, or those chosen with ``classes``)."""
    select = task.get("CLASS_SELECT")
    if select is None or n_classes == 1:
        return range(n_classes)
    return [c for c in range(n_classes) if c in select]


def _file_stem(name, max_len=120):
    """
    File name (without extension) for a figure name: characters that are not
    allowed in file names are replaced, and a very long name is shortened
    (with a short hash, so that different long names stay different).
    """
    stem = re.sub(r"[^\w.+=,() -]", "_", name.replace(":", "__"))
    if len(stem) > max_len:
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        stem = stem[: max_len - 9] + "_" + digest
    return stem


def print_parameter_distributions(task, results, figs):
    """
    Print model parameter distributions in histogram.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    figs : _Figures
        Collector that receives the finished figures.

    Returns
    -------
    None.
    """

    # Prepare results ---------------------------------------------------------
    # Get params
    params = pd.DataFrame(results["best_params"])

    # Make plot ---------------------------------------------------------------
    # Iterate over columns of params dataframe
    for name, data in params.items():
        # Make figure
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot hist of inlier score
        _call(sns.histplot, figs.kwargs("parameters"), 
            data=data.astype("float"), bins=30, kde=True, color="#777777", ax=ax
        )
        # Remove top, right and left frame elements
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Add x label
        ax.set_xlabel(name)
        # Add y label
        ax.set_ylabel("Number")
        # Set title
        ax.set_title(
            task["ANALYSIS_NAME"]
            + " "
            + "parameter distribution for predicting"
            + " "
            + task["y_name"][0],
            fontsize=10,
        )

        # Save figure ---------------------------------------------------------
        # Make save path
        figs.add(f"parameters:{name}", "parameters", fig, ax)

    # Return None -------------------------------------------------------------
    return fig


def print_regression_scatter(task, results, figs):
    """
    Print model fit in a scatter plot (regression).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    figs : _Figures
        Collector that receives the finished figures.

    Returns
    -------
    None.
    """

    # Prepare results ---------------------------------------------------------
    # True values
    true_values = np.concatenate([i["y_true"] for i in results["scores"]])
    # Predicted values
    pred_values = np.concatenate([i["y_pred"] for i in results["scores"]])

    # Make plot ---------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Print data
    _call(ax.scatter, figs.kwargs("fit_scatter"), pred_values, true_values, zorder=2, alpha=0.1, color="#444444")
    # Add optimal fit line
    ax.plot(
        [-10000, 10000],
        [-10000, 10000],
        color="#999999",
        zorder=3,
        linewidth=2,
        alpha=0.3,
    )
    # Fix aspect
    ax.set_aspect(1)
    # Remove top, right and left frame elements
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Remove ticks
    ax.tick_params(
        axis="both",
        which="major",
        reset=True,
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    # Add grid
    ax.grid(visible=True, which="major", axis="both")
    # Modify grid
    ax.tick_params(grid_linestyle=":", grid_alpha=0.5)
    # Get true values range
    true_values_range = max(true_values) - min(true_values)
    # Set x-axis limits
    ax.set_xlim(
        min(true_values) - true_values_range / 20,
        max(true_values) + true_values_range / 20,
    )
    # Set y-axis limits
    ax.set_ylim(
        min(true_values) - true_values_range / 20,
        max(true_values) + true_values_range / 20,
    )
    # Set title
    ax.set_title(
        task["ANALYSIS_NAME"] + " " + "predicting" + " " + task["y_name"][0],
        fontsize=10,
    )
    # Set xlabel
    ax.set_xlabel("Predicted values", fontsize=10)
    # Set x ticks size
    plt.xticks(fontsize=10)
    # Set ylabel
    ax.set_ylabel("True values", fontsize=10)
    # Set y ticks size
    plt.yticks(fontsize=10)

    # Add MAE -----------------------------------------------------------------
    # Extract MAE
    mae = [i["mae"] for i in results["scores"]]
    # Extract MAE shuffle
    mae_sh = [i["mae"] for i in results["scores_sh"]]
    # Calculate p-value between MAE and shuffle MAE
    _, pval_mae = corrected_ttest(np.array(mae_sh) - np.array(mae))
    # Add MAE results to plot
    ax.text(
        0.40,
        0.055,
        (
            "MAE original mean"
            + r"$\pm$"
            + "std:{:.2f}"
            + r"$\pm$"
            + "{:.2f}|med:{:.2f}"
        ).format(np.mean(mae), np.std(mae), np.median(mae)),
        transform=ax.transAxes,
        fontsize=8,
    )
    # Add MAE p val results to plot
    ax.text(
        0.40,
        0.02,
        (
            "MAE shuffle mean"
            + r"$\pm$"
            + "std:{:.2f}"
            + r"$\pm$"
            + "{:.2f}|med:{:.2f}|{}"
        ).format(
            np.mean(mae_sh), np.std(mae_sh), np.median(mae_sh), format_p(pval_mae)
        ),
        transform=ax.transAxes,
        fontsize=8,
    )

    # Add R² ------------------------------------------------------------------
    # Extract R²
    r2 = [i["r2"] for i in results["scores"]]
    # Extract R² shuffle
    r2_sh = [i["r2"] for i in results["scores_sh"]]
    # Calculate p-value between R² and shuffle R²
    _, pval_r2 = corrected_ttest(np.array(r2) - np.array(r2_sh))
    # Add R² results to plot
    ax.text(
        0.02,
        0.96,
        (
            "R² original mean"
            + r"$\pm$"
            + "std:{:.3f}"
            + r"$\pm$"
            + "{:.3f}|med:{:.3f}"
        ).format(np.mean(r2), np.std(r2), np.median(r2)),
        transform=ax.transAxes,
        fontsize=8,
    )
    # Add R² p val results to plot
    ax.text(
        0.02,
        0.925,
        (
            "R² shuffle mean"
            + r"$\pm$"
            + "std:{:.3f}"
            + r"$\pm$"
            + "{:.3f}|med:{:.3f}|{}"
        ).format(np.mean(r2_sh), np.std(r2_sh), np.median(r2_sh), format_p(pval_r2)),
        transform=ax.transAxes,
        fontsize=8,
    )

    # Save figure -------------------------------------------------------------
    # Make save path
    figs.add("fit_scatter", "fit", fig, ax)

    # Return None -------------------------------------------------------------
    return fig


def print_regression_violin(task, results, figs):
    """
    Print model fit in a violin plot (regression).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    figs : _Figures
        Collector that receives the finished figures.

    Returns
    -------
    None.
    """

    # Prepare results ---------------------------------------------------------
    # Extract MAE
    mae = [i["mae"] for i in results["scores"]]
    # Extract MAE shuffle
    mae_sh = [i["mae"] for i in results["scores_sh"]]
    # Extract R²
    r2 = [i["r2"] for i in results["scores"]]
    # Extract R² shuffle
    r2_sh = [i["r2"] for i in results["scores_sh"]]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {
            "Mean Absolute Error": pd.Series(np.array(mae)),
            "R2": pd.Series(np.array(r2)),
            "Data": pd.Series(["original" for _ in mae]),
            "Dummy": pd.Series(np.ones(np.array(mae).shape).flatten()),
        }
    )
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {
            "Mean Absolute Error": pd.Series(np.array(mae_sh)),
            "R2": pd.Series(np.array(r2_sh)),
            "Data": pd.Series(["shuffle" for _ in mae_sh]),
            "Dummy": pd.Series(np.ones(np.array(mae_sh).shape).flatten()),
        }
    )
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ["Mean Absolute Error", "R2"]

    # Make plot ---------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, len(metrics) * 0.75 + 1))
    # Set tight figure layout
    fig.tight_layout()
    # Make color palette
    mypal = {"original": "#777777", "shuffle": "#eeeeee"}
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        _call(sns.violinplot, figs.kwargs("fit_violin"), 
            x=metric,
            y="Dummy",
            hue="Data",
            data=all_scores_df,
            bw_method="scott",
            cut=2,
            density_norm="width",
            gridsize=100,
            width=0.8,
            inner="box",
            orient="h",
            linewidth=1,
            saturation=1,
            ax=ax[i],
            palette=mypal,
        )
        # Remove top, right and left frame elements
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        # Remove ticks
        ax[i].tick_params(
            axis="both",
            which="major",
            reset=True,
            bottom=True,
            top=False,
            left=False,
            right=False,
            labelleft=False,
        )
        # Set x ticks and size
        ax[i].set_xlabel(metrics[i], fontsize=10)
        # Set y ticks and size
        ax[i].set_ylabel("", fontsize=10)
        # For other than first metric
        if i > 0:
            # Remove legend
            ax[i].legend().remove()
        # Add horizontal grid
        fig.axes[i].set_axisbelow(True)
        # Set grid style
        fig.axes[i].grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)
    # Make title string
    title_str = task["ANALYSIS_NAME"] + " " + "predicting" + " " + task["y_name"][0]
    # set title
    fig.axes[0].set_title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    figs.add("fit_violin", "fit", fig, ax)

    # Return None -------------------------------------------------------------
    return fig


def print_classification_confusion(task, results, figs):
    """
    Print model fit as confusion matrix (classification).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    figs : _Figures
        Collector that receives the finished figures.

    Returns
    -------
    None.
    """

    # Prepare results ---------------------------------------------------------
    # True values
    true_values = [i["y_true"] for i in results["scores"]]
    # Predicted values
    pred_values = [i["y_pred"] for i in results["scores"]]
    # Sample weights  list
    sample_weights = [i["class_weights"] for i in results["scores"]]
    # Accuracy
    acc = [i["acc"] for i in results["scores"]]
    # Schuffle accuracy
    acc_sh = [i["acc"] for i in results["scores_sh"]]
    # Get classes
    class_labels = np.unique(np.concatenate(true_values)).tolist()

    # Get confusion matrix ----------------------------------------------------
    # Loop over single results
    for true, pred, w in zip(true_values, pred_values, sample_weights):
        if "con_mat" not in locals():
            # Compute confusion matrix
            con_mat = confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=np.array([w[i] for i in true]),
                normalize="all",
            )
        else:
            # Add confusion matrix
            con_mat = np.add(
                con_mat,
                confusion_matrix(
                    true,
                    pred,
                    labels=class_labels,
                    sample_weight=np.array([w[i] for i in true]),
                    normalize="all",
                ),
            )
    # Normalize confusion matrix
    con_mat_norm = con_mat / len(true_values)

    # Plot confusion matrix ---------------------------------------------------
    # Create figure
    fig, ax = plt.subplots(
        figsize=(con_mat.shape[0] * 0.5 + 3, con_mat.shape[0] * 0.5 + 3)
    )
    # Plot confusion matrix
    _call(sns.heatmap, figs.kwargs("fit_confusion"), 
        con_mat_norm * 100,
        vmin=None,
        vmax=None,
        cmap="Greys",
        center=None,
        robust=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=1,
        linecolor="#999999",
        cbar=False,
        cbar_kws=None,
        square=True,
        xticklabels=class_labels,
        yticklabels=class_labels,
        mask=None,
        ax=ax,
    )
    # Add x label to plot
    plt.xlabel("Predicted class", fontsize=10)
    # Add y label to plot
    plt.ylabel("True class", fontsize=10)
    # Set y ticks size and sets the yticks 'upright' with 0
    plt.yticks(rotation=0, fontsize=10)
    # Calculate p-value of accuracy and shuffle accuracy
    _, pval_acc = corrected_ttest(np.array(acc) - np.array(acc_sh))
    # Make title string
    title_str = (
        task["ANALYSIS_NAME"]
        + " "
        + "predicting"
        + " "
        + task["y_name"][0]
        + "\n"
        + "Orig. data accuracy mean"
        + r"$\pm$"
        + "std|median: {:.2f}"
        + r"$\pm$"
        + "{:.2f}|{:.2f}"
        + "\n"
        + "Shuffle data accuracy mean"
        + r"$\pm$"
        + "std|median: {:.2f}"
        + r"$\pm$"
        + "{:.2f}|{:.2f}"
        + "\n"
        + "p-value of orig. and shuffle: {}"
        + "\n"
    ).format(
        np.mean(acc) * 100,
        np.std(acc) * 100,
        np.median(acc) * 100,
        np.mean(acc_sh) * 100,
        np.std(acc_sh) * 100,
        np.median(acc_sh) * 100,
        format_p(pval_acc),
    )
    # Set title
    plt.title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    figs.add("fit_confusion", "fit", fig, ax)

    # Return None -------------------------------------------------------------
    return fig


def print_classification_violin(task, results, figs):
    """
    Print model fit in a violin plot (classification).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    figs : _Figures
        Collector that receives the finished figures.

    Returns
    -------
    None.
    """

    # Prepare results ---------------------------------------------------------
    # Extract accuracy
    acc = [i["acc"] for i in results["scores"]]
    # Extract shuffle accuracy
    acc_sh = [i["acc"] for i in results["scores_sh"]]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {
            "Accuracy": pd.Series(np.array(acc)),
            "Data": pd.Series(["original" for _ in acc]),
            "Dummy": pd.Series(np.ones(np.array(acc).shape).flatten()),
        }
    )
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {
            "Accuracy": pd.Series(np.array(acc_sh)),
            "Data": pd.Series(["shuffle" for _ in acc_sh]),
            "Dummy": pd.Series(np.ones(np.array(acc_sh).shape).flatten()),
        }
    )
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ["Accuracy"]

    # Make plot ---------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(figsize=(8, len(metrics) * 0.75 + 1))
    # Make color palette
    mypal = {"original": "#777777", "shuffle": "#eeeeee"}
    # Put ax into list
    ax = [ax]
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        _call(sns.violinplot, figs.kwargs("fit_violin"), 
            x=metric,
            y="Dummy",
            hue="Data",
            data=all_scores_df,
            bw_method="scott",
            cut=2,
            density_norm="width",
            gridsize=100,
            width=0.8,
            inner="box",
            orient="h",
            linewidth=1,
            saturation=1,
            ax=ax[i],
            palette=mypal,
        )
        # Remove top, right and left frame elements
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        # Remove ticks
        ax[i].tick_params(
            axis="both",
            which="major",
            reset=True,
            bottom=True,
            top=False,
            left=False,
            right=False,
            labelleft=False,
        )
        # Set x ticks and size
        ax[i].set_xlabel(metrics[i], fontsize=10)
        # Set y ticks and size
        ax[i].set_ylabel("", fontsize=10)
        # For other than first metric
        if i > 0:
            # Remove legend
            ax[i].legend().remove()
        # Add horizontal grid
        fig.axes[i].set_axisbelow(True)
        # Set grid style
        fig.axes[i].grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)
    # Make title string
    title_str = task["ANALYSIS_NAME"] + " " + "predicting" + " " + task["y_name"][0]
    # set title
    plt.title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    figs.add("fit_violin", "fit", fig, ax)

    # Return None -------------------------------------------------------------
    return fig


def print_shap_effects(task, results, figs):
    """
    Print SHAP based global effects.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    figs : _Figures
        Collector that receives the finished figures.

    Returns
    -------
    None.
    """

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task["SHAP_INTERACTIONS"] and (
        task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
    ):
        # Get number of classes
        n_classes = results["explainations"][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap effects -------------------------------------------------------
    # Loop over classes
    for c_class in _class_loop(task, n_classes):
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(
            task, results["explainations"], c_class
        )
        # Get current shuffle shap effects
        shap_effects_sh_df, _ = get_shap_effects(
            task, results["explainations_sh"], c_class
        )

        # Process SHAP effects-------------------------------------------------
        # Mean shap values
        shap_effects_se_mean = shap_effects_df.mean(axis=0)
        # Sort from highto low
        shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(ascending=True)

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task["x_names"]])
        # x names count
        x_names_count = len(task["x_names"])

        # Plot ----------------------------------------------------------------
        # Make figure (own axes: pandas would otherwise draw on the current figure)
        fig, ax = plt.subplots(
            figsize=(x_names_max_len * 0.1 + 7, x_names_count * 0.4 + 1)
        )
        # Make horizontal bar plot
        _call(
            shap_effects_se_mean_sort.plot,
            figs.kwargs("shap_effects"),
            kind="barh",
            ax=ax,
            color="#777777",
            fontsize=10,
        )
        # Set x label size
        plt.xlabel("mean(|SHAP values|)", fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Remove top, right and left frame elements
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Add horizontal grid
        ax.set_axisbelow(True)
        # Set grid style
        ax.grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)
        # Make title string
        title_str = (
            task["ANALYSIS_NAME"]
            + " "
            + "SHAP effects for"
            + " "
            + task["y_name"][0]
            + "\n"
            + "mean(|SHAP values|) = mean absolute deviation from expected"
            + " value ("
            + str(np.round(base, decimals=2))
            + ")"
        )
        # Add class if no interactions and binary or multiclass
        if not task["SHAP_INTERACTIONS"] and (
            task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
        ):
            # Make title string
            title_str = title_str + " class: " + str(c_class)
        # Set title
        ax.set_title(title_str, fontsize=10)

        # Compute SHAP effect p values (shared with the estimator) ----------------
        pval_se = pd.Series(
            adjust_pvalues(
                effect_pvalues(
                    shap_effects_df.to_numpy(), shap_effects_sh_df.to_numpy()
                ),
                task["MCC"],
            ),
            index=task["x_names"],
        )

        # Add SHAP effect values and p values as text -------------------------
        # Loop over values
        for i, (c_pred, c_val) in enumerate(shap_effects_se_mean_sort.items()):
            # Make test string
            txt_str = (
                str(np.around(c_val, decimals=2))
                + "|"
                +
                #'p '+str(pval_se[c_pred]))
                format_p(pval_se[c_pred])
            )
            # Add values to plot
            ax.text(c_val, i, txt_str, color="k", va="center", fontsize=8)
        # Get x limits
        x_left, x_right = plt.xlim()
        plt.xlim(x_left, x_right + x_right * 0.1)

        # Save plot -----------------------------------------------------------
        # Make save path
        figs.add(_name(task, "shap_effects", c_class), "shap_effects", fig, ax)

    # Return None -------------------------------------------------------------
    return fig


def print_shap_effects_distribution(task, results, figs):
    """
    Print SHAP values distribution.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    figs : _Figures
        Collector that receives the finished figures.

    Returns
    -------
    None.
    """

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task["SHAP_INTERACTIONS"] and (
        task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
    ):
        # Get number of classes
        n_classes = results["explainations"][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap effects distribution ------------------------------------------
    # Loop over classes
    for c_class in _class_loop(task, n_classes):
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(
            task, results["explainations"], c_class
        )
        # Get current shuffle shap effects
        shap_effects_sh_df, _ = get_shap_effects(
            task, results["explainations_sh"], c_class
        )

        # Process SHAP effects-------------------------------------------------
        # Sorting index by mean value of columns
        i_srt = shap_effects_df.mean().sort_values(ascending=False).index
        # Sort SHAP effects dataframe
        shap_effects_df_sort = shap_effects_df.reindex(i_srt, axis=1)
        # Sort shuffle SHAP effects dataframe
        shap_effects_sh_df_sort = shap_effects_sh_df.reindex(i_srt, axis=1)
        # Add data origin to SHAP effects dataframe
        shap_effects_df_sort["Data"] = pd.DataFrame(
            ["original" for _ in range(shap_effects_df_sort.shape[0])], columns=["Data"]
        )
        # Add data origin to shuffle SHAP effects dataframe
        shap_effects_sh_df_sort["Data"] = pd.DataFrame(
            ["shuffle" for _ in range(shap_effects_sh_df_sort.shape[0])],
            columns=["Data"],
        )
        # Get value name
        value_name = "mean(|SHAP value|)"
        # Melt SHAP effects dataframe
        shap_effects_df_sort_melt = shap_effects_df_sort.melt(
            id_vars=["Data"], var_name="predictors", value_name=value_name
        )
        # Melt shuffle SHAP effects dataframe
        shap_effects_sh_df_sort_melt = shap_effects_sh_df_sort.melt(
            id_vars=["Data"], var_name="predictors", value_name=value_name
        )
        # Concatenate importances dataframes
        shap_effects_df_sort_melt_all = pd.concat(
            [shap_effects_df_sort_melt, shap_effects_sh_df_sort_melt], axis=0
        )

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task["x_names"]])
        # x names count
        x_names_count = len(task["x_names"])

        # Plot ----------------------------------------------------------------
        # Make figure
        fig, ax = plt.subplots(
            figsize=(x_names_max_len * 0.1 + 7, x_names_count * 0.4 + 1)
        )
        # Make color palette
        mypal = {"original": "#777777", "shuffle": "#eeeeee"}
        # Plot data
        _call(sns.violinplot, figs.kwargs("shap_effects_distribution"), 
            x=value_name,
            y="predictors",
            hue="Data",
            data=shap_effects_df_sort_melt_all,
            bw_method="scott",
            cut=2,
            density_norm="width",
            gridsize=100,
            width=0.8,
            inner="box",
            orient="h",
            linewidth=0.5,
            saturation=1,
            ax=ax,
            palette=mypal,
        )
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel("mean(|SHAP values|)", fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel("", fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Remove top, right and left frame elements
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Add horizontal grid
        ax.set_axisbelow(True)
        # Set grid style
        ax.grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)
        # Set legend position
        plt.legend(loc="lower right")
        # Make title string
        title_str = (
            task["ANALYSIS_NAME"]
            + " "
            + "SHAP effects distribution for"
            + " "
            + task["y_name"][0]
            + "\n"
            + "mean(|SHAP values|) = mean absolute deviation from expected"
            + " value ("
            + str(np.round(base, decimals=2))
            + ")"
        )
        # Add class if no interactions and binary or multiclass
        if not task["SHAP_INTERACTIONS"] and (
            task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
        ):
            # Make title string
            title_str = title_str + " class: " + str(c_class)
        # Add title
        ax.set_title(title_str, fontsize=10)

        # Save plots and results ----------------------------------------------
        # Make save path
        figs.add(_name(task, "shap_effects_distribution", c_class), "shap_effects_distribution", fig, ax)

    # Return None -------------------------------------------------------------
    return fig


def print_shap_values(task, results, figs):
    """
    Plot SHAP values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    figs : _Figures
        Collector that receives the finished figures.

    Returns
    -------
    None.
    """

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task["SHAP_INTERACTIONS"] and (
        task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
    ):
        # Get number of classes
        n_classes = results["explainations"][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in _class_loop(task, n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task, results["explainations"], c_class)

        # If interactions
        if task["SHAP_INTERACTIONS"]:
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
        x_names_max_len = max([len(i) for i in task["x_names"]])
        # x names count
        x_names_count = len(task["x_names"])

        # Plot SHAP values beeswarm -------------------------------------------
        _shap_call(beeswarm, figs.kwargs("shap_values"), 
            shap_explainations,
            max_display=len(task["x_names"]),
            order=Explanation.abs.mean(0),
            clustering=None,
            cluster_threshold=0.5,
            color=None,
            axis_color="#333333",
            alpha=0.66,
            show=False,
            log_scale=False,
            color_bar=True,
            plot_size=(x_names_max_len * 0.1 + 7, x_names_count * 0.4 + 1),
            color_bar_label="Predictor value",
        )
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel("SHAP values", fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Make title string
        title_str = (
            task["ANALYSIS_NAME"]
            + " "
            + "SHAP values for"
            + " "
            + task["y_name"][0]
            + "\n"
            + "mean(|SHAP values|) = mean absolute deviation from expected"
            + " value ("
            + str(np.round(base, decimals=2))
            + ")"
        )
        # Add class if no interactions and binary or multiclass
        if not task["SHAP_INTERACTIONS"] and (
            task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
        ):
            # Make title string
            title_str = title_str + " class: " + str(c_class)
        # Add title
        plt.title(title_str, fontsize=10)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel("Predictor value", fontsize=10)

        # Save plot -----------------------------------------------------------
        # Make save path
        figs.add(_name(task, "shap_values", c_class), "shap_values", fig, ax)

    # Return None -------------------------------------------------------------
    return fig


def print_shap_dependences(task, results, figs):
    """
    Plot SHAP dependences.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    figs : _Figures
        Collector that receives the finished figures.

    Returns
    -------
    None.
    """

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task["SHAP_INTERACTIONS"] and (
        task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
    ):
        # Get number of classes
        n_classes = results["explainations"][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in _class_loop(task, n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task, results["explainations"], c_class)
        # If interactions
        if task["SHAP_INTERACTIONS"]:
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
            _call(scatter, figs.kwargs("dependence"), 
                shap_explainations[:, i],
                color="#777777",
                hist=True,
                axis_color="#333333",
                dot_size=16,
                x_jitter="auto",
                alpha=0.5,
                title=None,
                xmin=None,
                xmax=None,
                ymin=None,
                ymax=None,
                overlay=None,
                ax=ax,
                ylabel="SHAP values",
                show=False,
            )
            # Get the current figure and axes objects.
            fig, ax = plt.gcf(), plt.gca()
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
                task["ANALYSIS_NAME"]
                + " "
                + "SHAP values for"
                + " "
                + task["y_name"][0]
                + "\n"
                + "SHAP values = deviation from expected value ("
                + str(np.round(base, decimals=2))
                + ")"
            )
            # Add class if no interactions and binary or multiclass
            if not task["SHAP_INTERACTIONS"] and (
                task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
            ):
                # Make title string
                title_str = title_str + " class: " + str(c_class)
            # Add title
            plt.title(title_str, fontsize=10)

            # Save plot -------------------------------------------------------
            # Make save path
            figs.add(_name(task, f"dependence:{c_pred}", c_class), "dependence", fig, ax)

    # Return None -------------------------------------------------------------
    return fig


def print_shap_effects_interactions(task, results, figs):
    """
    Plot SHAP effects inclusive interactions.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    figs : _Figures
        Collector that receives the finished figures.

    Returns
    -------
    None.
    """

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task["SHAP_INTERACTIONS"] and (
        task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
    ):
        # Get number of classes
        n_classes = results["explainations"][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap effects -------------------------------------------------------
    # Loop over classes
    for c_class in _class_loop(task, n_classes):
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(
            task, results["explainations"], c_class
        )

        # Process SHAP effects-------------------------------------------------
        # Mean shap values
        shap_effects_se_mean = shap_effects_df.mean(axis=0)
        # Sort from highto low
        shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(ascending=False)

        # Get SHAP effects interactions ---------------------------------------
        # SHAP effects
        shap_effects_inter = get_interaction_effects(
            task, results["explainations"], c_class
        )
        # Make dataframe
        shap_effects_inter_df = pd.DataFrame(
            np.mean(shap_effects_inter, axis=0),
            index=task["x_names"],
            columns=task["x_names"],
        )
        # Reindex to sorted index
        shap_effects_inter_sort_df = shap_effects_inter_df.reindex(
            shap_effects_se_mean_sort.index
        )
        # Reorder columns to sorted index
        shap_effects_inter_sort_df = shap_effects_inter_sort_df.loc[
            :, shap_effects_se_mean_sort.index
        ]
        # SHAP effects shuffle
        shap_effects_inter_sh = get_interaction_effects(
            task, results["explainations_sh"], c_class
        )

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task["x_names"]])
        # x names count
        x_names_count = len(task["x_names"])

        # Make labels with pvales ---------------------------------------------
        # Get p values (shared with the estimator); the family of tests is all
        # cells of the matrix
        pval = adjust_pvalues(
            effect_pvalues(shap_effects_inter, shap_effects_inter_sh), task["MCC"]
        )
        # Initialize labels dataframe
        interaction_labels_df = pd.DataFrame(
            np.zeros([shap_effects_inter.shape[1], shap_effects_inter.shape[2]]),
            dtype=object,
        )
        # Iterate labels
        for x, y in np.ndindex(
            (shap_effects_inter.shape[1], shap_effects_inter.shape[2])
        ):
            # Make label
            interaction_labels_df.iloc[x, y] = (
                str(np.around(shap_effects_inter_df.iloc[x, y], decimals=2))
                + "\n"
                + "p"
                + " "
                + str(np.around(pval[x, y], decimals=3))
            )
        # Index labels dataframe
        interaction_labels_df.index = shap_effects_inter_df.index
        # Column labels
        interaction_labels_df.columns = shap_effects_inter_df.columns
        # Reindex to sorted index
        interaction_labels_sort_df = interaction_labels_df.reindex(
            shap_effects_se_mean_sort.index
        )
        # Reorder columns to sorted index
        interaction_labels_sort_df = interaction_labels_sort_df.loc[
            :, shap_effects_se_mean_sort.index
        ]

        # Plot interaction effects --------------------------------------------
        # Create figure
        fig, ax = plt.subplots(
            figsize=(
                x_names_max_len * 0.1 + x_names_count * 1 + 1,
                x_names_max_len * 0.1 + x_names_count * 1 + 1,
            )
        )
        # Make colorbar string
        clb_str = "mean(|SHAP value|)"
        # Plot confusion matrix
        _call(sns.heatmap, figs.kwargs("interaction_effects"), 
            shap_effects_inter_sort_df,
            vmin=None,
            vmax=None,
            cmap="Greys",
            center=None,
            robust=True,
            annot=interaction_labels_sort_df,
            fmt="",
            annot_kws={"size": 10},
            linewidths=1,
            linecolor="#999999",
            cbar=True,
            cbar_kws={"label": clb_str, "shrink": 0.6},
            square=True,
            xticklabels=True,
            yticklabels=True,
            mask=None,
            ax=ax,
        )
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
            task["ANALYSIS_NAME"]
            + " "
            + "SHAP effects for"
            + " "
            + task["y_name"][0]
            + "\n"
            + "mean(|SHAP values|) = deviation from expected value ("
            + str(
                np.round(
                    np.mean(
                        np.hstack([k.base_values for k in results["explainations"]])
                    ),
                    decimals=2,
                )
            )
            + ")"
        )
        # Add class if no interactions and binary or multiclass
        if not task["SHAP_INTERACTIONS"] and (
            task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
        ):
            # Make title string
            title_str = title_str + " class: " + str(c_class)
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
        figs.add(_name(task, "interaction_effects", c_class), "interaction_effects", fig, ax)

    # Return None -------------------------------------------------------------
    return fig


def print_shap_interaction_values(task, results, figs):
    """
    Plot SHAP interaction values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    figs : _Figures
        Collector that receives the finished figures.

    Returns
    -------
    None.
    """

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task["SHAP_INTERACTIONS"] and (
        task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
    ):
        # Get number of classes
        n_classes = results["explainations"][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in _class_loop(task, n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task, results["explainations"], c_class)

        # Print shap values dependencies --------------------------------------
        # Make list of permutations
        permutations_list = [(i, i) for i in shap_values.feature_names] + list(
            permutations(shap_values.feature_names, 2)
        )
        # Loop over predictor pairs
        for ind in permutations_list:
            # Make figure
            fig, ax = plt.subplots(figsize=(8, 5))
            # Plot SHAP dependence
            _call(dependence_plot, figs.kwargs("interaction_values"), 
                ind,
                shap_values=shap_values.values,
                features=pd.DataFrame(
                    shap_values.data, columns=shap_values.feature_names
                ),
                feature_names=shap_values.feature_names,
                display_features=None,
                interaction_index="auto",
                color="#1E88E5",
                axis_color="#333333",
                cmap=None,
                dot_size=16,
                x_jitter=0,
                alpha=0.66,
                title=None,
                xmin=None,
                xmax=None,
                ax=ax,
                show=False,
                ymin=None,
                ymax=None,
            )
            # Get the current figure and axes objects.
            fig, ax = plt.gcf(), plt.gca()
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
                task["ANALYSIS_NAME"]
                + " "
                + "SHAP interaction values for"
                + " "
                + task["y_name"][0]
                + "\n"
                + "SHAP values = deviation from expected value ("
                + str(
                    np.round(
                        np.mean(
                            np.hstack([k.base_values for k in results["explainations"]])
                        ),
                        decimals=2,
                    )
                )
                + ")"
            )
            # Add class if no interactions and binary or multiclass
            if not task["SHAP_INTERACTIONS"] and (
                task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
            ):
                # Make title string
                title_str = title_str + " class: " + str(c_class)
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
            figs.add(_name(task, f"interaction_values:{ind[0]}:{ind[1]}", c_class), "interaction_values", fig, ax)

    # Return None -------------------------------------------------------------
    return fig


def ilm_shap_values_graph(
    task,
    results,
    figs,
    title=None,
    exclude=None,
    rename_dict=None,
    stars=False,
):
    """
    Alternative graph of SHAP values (``shap_values_style='graph'``): a
    beeswarm plot with effect sizes and p values added to the labels.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    figs : _Figures
        Collector that receives the finished figures.
    title : str or None, default None
        Title of the figure.
    exclude : list of str or None, default None
        Predictors to leave out.
    rename_dict : dict or None, default None
        Predictor name -> label shown in the figure.
    stars : bool, default False
        Use significance stars instead of p values.

    These four options can also be given through
    ``plot_kwargs={'shap_values': {...}}`` of :func:`make_figures`; all other
    entries there go to the beeswarm plot.

    Returns
    -------
    None.
    """
    # Own options may come from plot_kwargs; the rest goes to the beeswarm plot
    user = figs.kwargs("shap_values")
    title = user.pop("title", title)
    exclude = user.pop("exclude", exclude) or []
    rename_dict = user.pop("rename_dict", rename_dict) or {}
    stars = user.pop("stars", stars)
    beeswarm_kwargs = user
    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task["SHAP_INTERACTIONS"] and (
        task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
    ):
        # Get number of classes
        n_classes = results["explainations"][0].shape[2]
    # Other cases
    else:
        n_classes = 1
    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in _class_loop(task, n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task, results["explainations"], c_class)
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(
            task, results["explainations"], c_class
        )
        # Get current shuffle shap effects
        shap_effects_sh_df, _ = get_shap_effects(
            task, results["explainations_sh"], c_class
        )
        # Process SHAP effects-------------------------------------------------
        # Mean shap values
        shap_effects_se_mean = shap_effects_df.mean(axis=0)
        # Sort from highto low
        shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(ascending=True)
        # If interactions
        if task["SHAP_INTERACTIONS"]:
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
        x_names_max_len = max([len(i) for i in task["x_names"]])
        # x names count
        # Compute SHAP effect p values (shared with the estimator) ----------------
        raw_pval = pd.Series(
            effect_pvalues(shap_effects_df.to_numpy(), shap_effects_sh_df.to_numpy()),
            index=task["x_names"],
        )
        excl = 0
        for e in exclude:
            idx = shap_explainations.feature_names.index(e)
            shap_explainations.feature_names.remove(e)
            shap_explainations.values = np.delete(
                shap_explainations.values, idx, axis=1
            )
            shap_explainations.data = np.delete(shap_explainations.data, idx, axis=1)
            excl += 1
            print(f"'{e}' has been removed.")
        # Adjust for multiple comparisons within the predictors that are shown
        x_names_count = len(task["x_names"])
        shown = [n for n in task["x_names"] if n not in exclude]
        pval_se = raw_pval.copy()
        pval_se[shown] = adjust_pvalues(raw_pval[shown].to_numpy(), task["MCC"])
        extra_x_names = {}
        extra_x_names_max_len = []
        for i, (c_pred, c_val) in enumerate(shap_effects_se_mean_sort.items()):
            # Make test string
            txt_str = (
                c_pred
                + "  "
                + f"{c_val:6<.2f}"
                + f"{', ' + format_p(pval_se[c_pred]) if not stars else get_stars(pval_se[c_pred])}"
            )
            extra_x_names[c_pred] = txt_str
        for i, f in enumerate(shap_explainations.feature_names):
            shap_explainations.feature_names[i] = (
                extra_x_names[f].replace(f, rename_dict[f])
                if f in rename_dict
                else extra_x_names[f]
            )
            extra_x_names_max_len.append(
                len(shap_explainations.feature_names[i].split("\n")[0])
            )
        extra_x_names_max_len = max(extra_x_names_max_len)

        # Plot SHAP values beeswarm -------------------------------------------
        _shap_call(beeswarm, beeswarm_kwargs, 
            shap_explainations,
            max_display=len(task["x_names"]),
            order=Explanation.abs.mean(0),
            clustering=None,
            cluster_threshold=0.5,
            color=None,
            axis_color="#333333",
            alpha=0.66,
            show=False,
            log_scale=False,
            color_bar=True,
            plot_size=(x_names_max_len * 0.05 + 7, x_names_count * 0.4 + 1),
            color_bar_label="Predictor value",
        )
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel("Impact on model output\n(SHAP value of predictor)", fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Make title string
        title_str = (
            task["ANALYSIS_NAME"]
            + " "
            + "SHAP values for"
            + " "
            + task["y_name"][0]
            + "\n"
            + "mean(|SHAP values|) = mean absolute deviation from expected"
            + " value ("
            + str(np.round(base, decimals=2))
            + ")"
        )
        # Add class if no interactions and binary or multiclass

        if not (task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"):
            # Extract R²
            r2 = [i["r2"] for i in results["scores"]]
            # Extract R² shuffle
            r2_sh = [i["r2"] for i in results["scores_sh"]]
            # Calculate p-value between R² and shuffle R²
            _, pval_r2 = corrected_ttest(np.array(r2) - np.array(r2_sh))
            # Add R² results to plot
            metrics_str = (
                f"R²pred={format_r(np.mean(r2))}"
                + r"±"
                + f"{format_p(np.std(r2), use_letter=False, keep_spaces=False, no_equals=True)}, {format_p(pval_r2, keep_spaces=False)}"
            )
        else:
            metrics_str = "Acc.="

        if not task["SHAP_INTERACTIONS"] and (
            task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass"
        ):
            # Make title string
            title_str = title_str + " class: " + str(c_class)
        # Add title
        print(title_str)
        print(metrics_str)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel("Predictor value", fontsize=10)
        # Add SHAP effect values and p values as text -------------------------
        # Loop over values
        x_left, x_right = plt.xlim()
        y_bottom, y_top = plt.ylim()
        # for i, (c_pred, c_val) in enumerate(shap_effects_se_mean_sort.items()):
        # Make test string
        #    txt_str = (str(np.around(c_val, decimals=2))+', '+format_p(pval_se[c_pred])) #get_stars(pval_se[c_pred])
        # Add values to plot
        #    ax.text(x_left-.15, i-.05, txt_str, color='k', va='center', fontsize=9) #c_val
        # Get x limits
        s = "Average impact   \nPredictor     (mean|SHAP values|)"
        ax.text(
            x_left - x_left * 0.05,
            y_top,
            s,
            color="k",
            va="center",
            ha="right",
            fontsize=10,
        )  # - (max([len(i) for i in s.split('\n')])*.02 / 2)
        plt.xlim(x_left - x_left * 0.05, x_right + x_right * 0.1)

        if title != None:
            plt.title(title, fontsize=10)
        else:
            f = task["y_name"][0]
            f = f.replace(f, rename_dict[f]) if f in rename_dict else f
            f += f"\n{metrics_str}"
            ax.text(
                (x_left + x_right) / 2,
                y_top + len(f.split("\n")) * 0.4,
                f,
                fontsize=10.6,
                horizontalalignment="center",
                verticalalignment="top",
            )

        # Save plot -----------------------------------------------------------
        # Make save path
        figs.add(_name(task, "shap_values", c_class), "shap_values", fig, ax)
    # Return None -------------------------------------------------------------
    return fig


def _class_results(results, i_class):
    """Results with the SHAP explanations reduced to one class (last axis)."""

    def one(e):
        return Explanation(
            np.asarray(e.values)[..., i_class],
            base_values=np.asarray(e.base_values)[..., i_class],
            data=e.data,
            feature_names=e.feature_names,
            compute_time=e.compute_time,
        )

    out = dict(results)
    out["explainations"] = [one(e) for e in results["explainations"]]
    out["explainations_sh"] = [one(e) for e in results["explainations_sh"]]
    return out


def _shap_figures(task, results, figs, types, style):
    """Draw the requested SHAP based figures of one task/results pair."""
    if "shap_effects" in types:
        print_shap_effects(task, results, figs)
    if "shap_effects_distribution" in types:
        print_shap_effects_distribution(task, results, figs)
    if "shap_values" in types:
        if style == "beeswarm":
            print_shap_values(task, results, figs)
        else:
            ilm_shap_values_graph(task, results, figs)
    if "dependence" in types:
        print_shap_dependences(task, results, figs)
    if "interaction_effects" in types:
        print_shap_effects_interactions(task, results, figs)
    if "interaction_values" in types:
        print_shap_interaction_values(task, results, figs)


def _check_plot_types(plot_type, interactions):
    """Validate ``plot_type``; return the types in drawing order."""
    if plot_type is None:
        types = list(DEFAULT_PLOT_TYPES)
    elif isinstance(plot_type, str) and plot_type == "all":
        types = [t for t in PLOT_TYPES if interactions or t not in INTERACTION_PLOT_TYPES]
    else:
        types = [plot_type] if isinstance(plot_type, str) else list(plot_type)
        unknown = [t for t in types if t not in PLOT_TYPES]
        if unknown:
            raise ValueError(
                f"Unknown plot_type {unknown}. Choose from {PLOT_TYPES} or 'all'."
            )
        missing = [t for t in types if t in INTERACTION_PLOT_TYPES and not interactions]
        if missing:
            raise ValueError(
                f"{missing} need SHAP interaction values: fit the estimator "
                "with shap_interactions=True."
            )
    return [t for t in PLOT_TYPES if t in types]


def _check_correction(correction):
    """Validate ``multiple_comparison_correction``; return method or None."""
    if correction is None or correction is False:
        return None
    if correction is True:
        return "bonferroni"
    if correction in CORRECTIONS:
        return correction
    raise ValueError(
        "multiple_comparison_correction must be None, True or one of "
        f"{CORRECTIONS}, got {correction!r}."
    )


def _check_figsize(figsize):
    """Validate ``figsize`` (None, a pair, or a dict plot type -> pair)."""

    def pair(x):
        try:
            w, h = x
            return float(w) > 0 and float(h) > 0
        except (TypeError, ValueError):
            return False

    if figsize is None or (not isinstance(figsize, dict) and pair(figsize)):
        return
    if isinstance(figsize, dict):
        bad = [k for k, v in figsize.items() if k not in PLOT_TYPES or not pair(v)]
        if not bad:
            return
        raise ValueError(
            f"figsize entries {bad} are invalid: keys must be in {PLOT_TYPES} "
            "and values (width, height) in inches."
        )
    raise ValueError("figsize must be None, (width, height) or a dict of those.")


def make_figures(
    model,
    plot_type=None,
    figsize=None,
    plot_kwargs=None,
    output_dir=None,
    fig_format="png",
    dpi=150,
    classes=None,
    shap_values_style="beeswarm",
    multiple_comparison_correction=None,
):
    """
    Draw figures of a fitted estimator and return them for further editing.

    Parameters
    ----------
    model : NCVGBDTRegressor, NCVGBDTClassifier, str or path
        A fitted estimator, or the directory of a stored one (see
        :func:`ncvgbdt.load_model`).
    plot_type : str, list of str or None, default None
        Which figures to draw. ``None``: ``'fit'``, ``'shap_effects'``,
        ``'shap_effects_distribution'`` and ``'shap_values'``. ``'all'``:
        every type that applies. Types:

        * ``'fit'``: regression: predicted vs. true (``fit_scatter``) and
          violin plots of R2 / MAE against the shuffled-target baseline
          (``fit_violin``); classification: class-weighted confusion matrix
          (``fit_confusion``) and violin plot of the balanced accuracy.
        * ``'shap_effects'``: mean absolute SHAP value per predictor with
          effect sizes and p values against the shuffled-target model.
        * ``'shap_effects_distribution'``: the same as violin plots over the
          outer folds, real vs. shuffled model.
        * ``'shap_values'``: SHAP beeswarm plot (or the alternative graph, see
          ``shap_values_style``).
        * ``'dependence'``: one SHAP dependence plot per predictor (many
          figures).
        * ``'interaction_effects'`` and ``'interaction_values'``: pairwise
          interaction effects and one interaction dependence plot per ordered
          pair of predictors (many figures); need ``shap_interactions=True``.
        * ``'parameters'``: distributions of the tuned hyperparameters.

        For classification, per-class types are drawn once per class (for
        binary targets class 0 is the mirror image of class 1; for multiclass
        targets with interaction values all SHAP types are drawn per class).
    figsize : tuple, dict or None, default None
        ``None``: default sizes, which depend on the number of predictors,
        classes and metrics. A ``(width, height)`` pair in inches is applied
        to every figure; a dict maps plot types to such pairs (types not
        listed keep their default).
    plot_kwargs : dict or None, default None
        Keyword arguments for the drawing call of a figure kind; they
        override the defaults. Keys (figure kinds) and the call that receives
        them: ``'fit_scatter'`` (``Axes.scatter``), ``'fit_violin'`` and
        ``'shap_effects_distribution'`` (``seaborn.violinplot``),
        ``'fit_confusion'`` and ``'interaction_effects'``
        (``seaborn.heatmap``), ``'shap_effects'`` (``Series.plot``),
        ``'shap_values'`` (``shap.plots.beeswarm``; with
        ``shap_values_style='graph'`` also ``title``, ``exclude``,
        ``rename_dict`` and ``stars``, see :func:`ilm_shap_values_graph`),
        ``'dependence'`` (``shap.plots.scatter``), ``'interaction_values'``
        (``shap.dependence_plot``) and ``'parameters'``
        (``seaborn.histplot``). Everything else can be changed afterwards on
        the returned figures.
    output_dir : str, path or None, default None
        Directory in which to save the figures (see :func:`save_figures`).
        ``None``: nothing is written.
    fig_format : {'png', 'svg', 'pdf'} or list of them, default 'png'
        File format(s) used if ``output_dir`` is given.
    dpi : int, default 150
        Resolution of PNG files (and of raster parts of other formats).
    classes : list or None, default None
        Classification only: labels of the classes to draw for per-class
        figures. ``None``: all classes.
    shap_values_style : {'beeswarm', 'graph'}, default 'beeswarm'
        ``'beeswarm'``: standard SHAP beeswarm plot. ``'graph'``: alternative
        graph with effect sizes and p values (:func:`ilm_shap_values_graph`).
    multiple_comparison_correction : {None, 'bonferroni', 'holm', 'fdr_bh'} or bool, default None
        Correction of the p values of the SHAP effects shown in the figures:
        Bonferroni, Holm (step-down, family-wise error rate) or
        Benjamini-Hochberg (false discovery rate). ``True`` means
        ``'bonferroni'``, ``None``/``False`` no correction. The family is the
        predictors of a figure (for interaction effects all cells of the
        matrix). The same p values are available as columns of
        ``effect_sizes_`` and ``interaction_effects_`` of the estimator.

    Returns
    -------
    dict
        ``{name: (fig, ax)}`` in drawing order. ``fig`` is a matplotlib
        ``Figure``; ``ax`` is its main ``Axes`` (for figures with several
        rows, the array of ``Axes``). Names are stable, e.g. ``'fit_scatter'``,
        ``'shap_values'``, ``'shap_effects:class_1'``, ``'dependence:wt'``,
        ``'interaction_values:a:b:class_0'``. The p values and effect sizes
        in the figures come from a one-sided corrected resampled t-test
        (Nadeau & Bengio, 2003), see the README for caveats.

    Raises
    ------
    sklearn.exceptions.NotFittedError
        If the estimator is not fitted.
    ValueError
        For an unknown ``plot_type``, interaction types without interaction
        values, or invalid ``figsize``, ``plot_kwargs``, ``fig_format``,
        ``dpi`` or ``classes``.

    Notes
    -----
    Figures are drawn with pyplot and stay open until you close them;
    ``plot_type='all'`` can create dozens. Close them with
    ``plt.close('all')`` (or ``plt.close(fig)``) when done. Edit a figure and
    save it again with :func:`save_figures`.
    """

    # Get the fitted estimator ------------------------------------------------
    if isinstance(model, (str, os.PathLike)):
        from .estimator import load_model

        model = load_model(model)
    if not hasattr(model, "results_"):
        from sklearn.exceptions import NotFittedError

        raise NotFittedError("Fit the estimator before plotting.")

    # Check arguments (before anything is drawn) --------------------------------
    task = dict(model.task_)
    interactions = task["SHAP_INTERACTIONS"]
    types = _check_plot_types(plot_type, interactions)
    _check_figsize(figsize)
    if plot_kwargs is not None:
        bad = [
            k
            for k, v in plot_kwargs.items()
            if k not in PLOT_KWARGS_KINDS or not isinstance(v, dict)
        ]
        if bad:
            raise ValueError(
                f"plot_kwargs entries {bad} are invalid: keys must be in "
                f"{PLOT_KWARGS_KINDS} and values dicts."
            )
    _check_correction(multiple_comparison_correction)
    if shap_values_style not in ("beeswarm", "graph"):
        raise ValueError("shap_values_style must be 'beeswarm' or 'graph'.")
    _check_format(fig_format, dpi)
    labels = list(model.classes_) if hasattr(model, "classes_") else None
    if classes is not None:
        if labels is None:
            raise ValueError("classes can only be used for classification.")
        unknown = [c for c in classes if c not in labels]
        if unknown:
            raise ValueError(f"Unknown classes {unknown}; the classes are {labels}.")
    select = None if classes is None else [labels.index(c) for c in classes]

    # Task for the figure functions -------------------------------------------------
    task["MCC"] = _check_correction(multiple_comparison_correction)
    task["CLASS_LABELS"] = labels
    task["CLASS_SELECT"] = select
    results = model.results_
    figs = _Figures(figsize, plot_kwargs)

    # Draw ----------------------------------------------------------------------
    if "parameters" in types:
        print_parameter_distributions(task, results, figs)
    if "fit" in types:
        if task["OBJECTIVE"] == "regression":
            print_regression_scatter(task, results, figs)
            print_regression_violin(task, results, figs)
        else:
            print_classification_confusion(task, results, figs)
            print_classification_violin(task, results, figs)
    shap_types = [t for t in types if t not in ("fit", "parameters")]
    if shap_types and task["OBJECTIVE"] == "multiclass" and interactions:
        # Interaction values exist per class. The interaction figures expect
        # one (rows x features x features) array, so draw the SHAP figures
        # for every class separately, like for a binary target.
        for i_class, label in enumerate(labels):
            if select is not None and i_class not in select:
                continue
            task_c = dict(
                task,
                OBJECTIVE="binary",
                y_name=[f"{task['y_name'][0]}_class_{label}"],
                NAME_SUFFIX=f":class_{label}",
            )
            _shap_figures(
                task_c, _class_results(results, i_class), figs, shap_types,
                shap_values_style,
            )
    elif shap_types:
        _shap_figures(task, results, figs, shap_types, shap_values_style)

    # Save and return ----------------------------------------------------------------
    if output_dir is not None:
        save_figures(figs.figures, output_dir, fig_format, dpi)
    return figs.figures


def _check_format(fig_format, dpi):
    """Validate file format(s) and resolution; return the formats as list."""
    formats = [fig_format] if isinstance(fig_format, str) else list(fig_format)
    bad = [f for f in formats if f not in FIG_FORMATS]
    if bad or not formats:
        raise ValueError(f"fig_format must be one or more of {FIG_FORMATS}, got {fig_format!r}.")
    if not isinstance(dpi, (int, float)) or isinstance(dpi, bool) or dpi <= 0:
        raise ValueError("dpi must be a positive number.")
    return formats


def save_figures(figures, output_dir, fig_format="png", dpi=150):
    """
    Save figures (e.g. from :func:`make_figures`, possibly edited) as files.

    Parameters
    ----------
    figures : dict
        ``{name: (fig, ax)}`` as returned by :func:`make_figures`.
    output_dir : str or path
        Directory (created if missing). Files are named after the figure
        names; characters that file names do not allow are replaced
        (``':'`` becomes ``'__'``), e.g. ``dependence__wt__class_1.png``.
        Existing files of the same name are overwritten, so use one
        directory per estimator.
    fig_format : {'png', 'svg', 'pdf'} or list of them, default 'png'
        File format(s).
    dpi : int, default 150
        Resolution of PNG files (and of raster parts of other formats).

    Returns
    -------
    list of str
        Paths of the files written.
    """
    formats = _check_format(fig_format, dpi)
    os.makedirs(output_dir, exist_ok=True)
    paths, used = [], set()
    for name, (fig, _) in figures.items():
        base = _file_stem(name)
        stem, k = base, 2
        while stem in used:  # names that only differ in replaced characters
            stem, k = f"{base}_{k}", k + 1
        used.add(stem)
        for fmt in formats:
            path = os.path.join(output_dir, f"{stem}.{fmt}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            paths.append(path)
    return paths
