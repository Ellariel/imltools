"""
Gradient Boosted Decision Trees (GBDT, LightGBM) with Nested Cross-Validation
(NCV) and SHAP (SHapley Additive exPlanations) -- model fitting.

Building blocks of the estimators in :mod:`ncvgbdt.estimator`
(:class:`~ncvgbdt.NCVGBDTRegressor`, :class:`~ncvgbdt.NCVGBDTClassifier`):
group k-fold splitters, the modelling pipeline, hyperparameter search,
scoring and SHAP explanations. Figures are produced by
:func:`ncvgbdt.plotting.make_figures`.

This code is a reused version with modifications by Danila Valko
(danila.valko[at]proton.me) of Dr. David Steyrl's codebase
(david.steyrl[at]gmail.com). Source code and description:
https://github.com/univiemops/iml, an example application:
https://doi.org/10.31219/osf.io/um69t. The SHAP project:
https://github.com/shap/shap.

You can cite this tool by mentioning one of Dr. Steyrl's works, e.g.
https://doi.org/10.1038/s41598-024-65088-z or
https://doi.org/10.1371/journal.pone.0304285

Method summary
--------------
* Outer loop ("CV"): repeated, group-wise 5-fold cross-validation
  (``n_rep_outer_cv`` repetitions). Groups (e.g. participants) are never split
  across train and test sets. Alternatively a single group-disjoint
  train-test split ("TT").
* Inner loop: randomized hyperparameter search on the training data only
  (5-fold group CV, repeated until about ``n_samples_inner_cv`` predictions;
  1 to 10 repetitions). Searched: column sample per tree (0.5 to 1), extra
  trees (True/False), path smoothing (1 to 1000, log scale). Fixed:
  learning rate 0.01, 100 leaves, 1000 boosting rounds, GOSS sampling.
* Regression is scored with R2, MAE and MSE; classification with balanced
  accuracy (inverse-frequency class weights).
* Every outer fold is repeated with shuffled training targets
  (permutation baseline, see README).
* SHAP values (tree explainer, optionally with interaction values) are
  computed on a subsample of each outer test fold.
"""

import os
import warnings
from collections import Counter
from time import time

import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from scipy.stats import loguniform
from scipy.stats import t
from scipy.stats import uniform
from shap import Explanation
from shap.explainers import Tree as TreeExplainer
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.utils import shuffle
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_random_state
from tqdm import tqdm

warnings.filterwarnings("ignore", "Setting an item of incompatible dtype is")

# Default seed. It equals the constant that was hard-coded for the outer CV
# split and the SHAP subsample in earlier versions.
DEFAULT_SEED = 3141592

################################################ KFold objects


# Sklearn compatible Repeated Group KFold Cross-Validation
# v1 Source: https://github.com/BbChip0103/sklearn_repeated_group_k_fold/blob/
# main/sklearn_repeated_group_k_fold.py
# @author: Dr. David Steyrl david.steyrl@gmail.com
class GroupKFold(_BaseKFold):
    """
    K-fold iterator variant with non-overlapping groups.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    The folds are approximately balanced in the sense that the number of
    distinct groups is approximately the same in each fold.
    Read more in the :ref:`User Guide <group_k_fold>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupKFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 0, 2, 2])
    >>> groups = np.array([0, 1, 2, 3])
    >>> group_kfold = GroupKFold(n_splits=2, shuffle=True, random_state=12345)
    >>> group_kfold.get_n_splits(X, y, groups)
    2
    >>> print(group_kfold)
    GroupKFold(n_splits=2, random_state=12345, shuffle=True)
    >>> for train_index, test_index in group_kfold.split(X, y, groups):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    ...    print(X_train, X_test, y_train, y_test)
    ...
    TRAIN: [0 1] TEST: [2 3]
    [[1 2]
     [3 4]] [[5 6]
     [7 8]] [0 0] [2 2]
    TRAIN: [2 3] TEST: [0 1]
    [[5 6]
     [7 8]] [[1 2]
     [3 4]] [2 2] [0 0]

    See Also
    --------
    LeaveOneGroupOut : For splitting the data according to explicit
        domain-specific stratification of the dataset.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)

        unique_groups, groups = np.unique(np.ravel(groups), return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )

        # Weight groups by their number of occurrences
        n_samples_per_group = np.bincount(groups)

        # Distribute the most frequent groups first
        indices = np.argsort(n_samples_per_group)[::-1]

        if self.shuffle:
            rng = check_random_state(self.random_state)
            for n_sample in np.unique(n_samples_per_group):
                same_n_indices_index = np.where(n_samples_per_group == n_sample)[0]
                target_chunk = indices[same_n_indices_index]
                rng.shuffle(target_chunk)
                indices[same_n_indices_index] = target_chunk

        n_samples_per_group = n_samples_per_group[indices]

        # Total weight of each fold
        n_samples_per_fold = np.zeros(self.n_splits)

        # Mapping from group index to fold index
        group_to_fold = np.zeros(len(unique_groups))

        # Distribute samples by adding the largest weight to the lightest fold
        for group_index, weight in enumerate(n_samples_per_group):
            lightest_fold = np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_index]] = lightest_fold

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        return super().split(X, y, groups)


class RepeatedGroupKFold(_RepeatedSplits):
    """
    Repeated Group K-Fold cross validator. Repeats Group K-Fold n times with
    different randomization in each repetition.
    Read more in the :ref:`User Guide <repeated_group_k_fold>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of each repeated cross-validation instance.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedGroupKFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 0, 2, 2])
    >>> groups = np.array([0, 1, 2, 3])
    >>> rkf = RepeatedGroupKFold(n_splits=2, n_repeats=2, random_state=2652124)
    >>> for train_index, test_index in rkf.split(X, y, groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...
    TRAIN: [1 2] TEST: [0 3]
    TRAIN: [0 3] TEST: [1 2]
    TRAIN: [0 1] TEST: [2 3]
    TRAIN: [2 3] TEST: [0 1]

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.

    See Also
    --------
    RepeatedStratifiedKFold : Repeats Stratified K-Fold n times.
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            GroupKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )

    def split(self, X, y=None, groups=None):
        """
        Generates indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        n_repeats = self.n_repeats
        rng = check_random_state(self.random_state)

        for idx in range(n_repeats):
            cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index


################################################# analysis
#################################################
#################################################


def create_dir(path):
    """
    Create specified directory if not existing.

    Parameters
    ----------
    path : string
        Path to to check to be created.

    Returns
    -------
    None.
    """

    # Create dir of not existing ----------------------------------------------
    # Check if dir exists
    os.makedirs(path, exist_ok=True)

    # Return None -------------------------------------------------------------
    return


def get_seed(task, offset=0, default=None):
    """
    Derive a random seed from the task seed.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    offset : int
        Offset added to the task seed, e.g. the outer CV iteration, so that
        different loops get different but reproducible seeds.
    default : int or None
        Value returned if no seed is set (task['SEED'] is None).

    Returns
    -------
    seed : int or None
        Seed to pass as random_state.
    """
    if task.get("SEED") is None:
        return default
    return task["SEED"] + offset


def get_te_target_type(objective):
    """
    Map the task objective to the ``target_type`` of the target encoder.

    Parameters
    ----------
    objective : str
        One of 'regression', 'binary', 'multiclass'.

    Returns
    -------
    target_type : str
        'continuous' for regression, 'binary' for binary classification and
        'multiclass' for multiclass classification. A multiclass encoder
        returns one column per class for every encoded predictor (named
        ``<predictor>_<class>``); these columns are summed back into one SHAP
        value per predictor in :func:`get_explainations`.
    """
    try:
        return {
            "regression": "continuous",
            "binary": "binary",
            "multiclass": "multiclass",
        }[objective]
    except KeyError:
        raise ValueError(
            "objective must be 'regression', 'binary' or "
            f"'multiclass', got {objective!r}."
        ) from None


def make_target_encoder(task):
    """
    Target encoder for the multi-category predictors.

    Uses 5-fold cross-fitting with shuffling, seeded by ``task['SEED']``. From
    scikit-learn 1.9 on, shuffling and seed are given to the encoder as a
    splitter object (``shuffle`` and ``random_state`` are deprecated there);
    older versions only accept the number of folds plus those two arguments.
    Both give the same folds.
    """
    objective = task["OBJECTIVE"]
    kwargs = dict(
        categories=task["te_categories"],
        target_type=get_te_target_type(objective),
        smooth="auto",
    )
    seed = get_seed(task)
    version = tuple(int(v) for v in sklearn.__version__.split(".")[:2])
    if version >= (1, 9):
        splitter = StratifiedKFold if objective != "regression" else KFold
        return TargetEncoder(
            cv=splitter(n_splits=5, shuffle=True, random_state=seed), **kwargs
        )
    return TargetEncoder(cv=5, shuffle=True, random_state=seed, **kwargs)


def prepare(task):
    """
    Prepare the analysis pipeline and the hyperparameter search space.

    Pipeline: (1) column transformer passing continuous and binary predictors
    on unchanged and target-encoding the multi-category predictors (target
    type follows the objective), (2) standardization of all columns, (3)
    LightGBM. Regression targets are standardized as well
    (``TransformedTargetRegressor``); SHAP values are transformed back to the
    original target units in :func:`get_explainations`.

    Fixed LightGBM settings: gbdt, 100 leaves, learning rate 0.01, 1000
    rounds, GOSS sampling (top_rate 0.5), min_child_samples 2, no L1/L2
    regularization, LightGBM default ``max_bin`` (255), ``min_data_in_bin`` 1. Class
    weights of classifiers are set per fold in :func:`tune_pipe`.

    Searched (random search): ``colsample_bytree`` ~ U(0.5, 1),
    ``extra_trees`` in {True, False}, ``path_smooth`` ~ log-U(1, 1000).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.

    Returns
    -------
    pipe : scikit-learn compatible analysis pipeline
        Prepared pipe object.
    space : dict
        Space that should be searched for optimale parameters.
    """

    # Make preprocessing pipe -------------------------------------------------
    # Instatiate target-encoder
    te = make_target_encoder(task)
    # Get categorical predictors for target-encoder
    coltrans = ColumnTransformer(
        [
            ("con_pred", "passthrough", task["X_CON_NAMES"]),
            ("bin_pred", "passthrough", task["X_CAT_BIN_NAMES"]),
            ("mult_pred", te, task["X_CAT_MULT_NAMES"]),
        ],
        remainder="drop",
        sparse_threshold=0,
        n_jobs=1,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=False,
    )
    # Pipeline
    pre_pipe = Pipeline(
        [("coltrans", coltrans), ("std_scaler", StandardScaler())],
        memory=None,
        verbose=False,
    )

    # Make predictor ----------------------------------------------------------
    # Regression
    if task["OBJECTIVE"] == "regression":
        # Estimator
        estimator = LGBMRegressor(
            boosting_type="gbdt",
            num_leaves=100,
            max_depth=-1,
            learning_rate=0.01,
            n_estimators=1000,
            subsample_for_bin=100000,
            objective=task["OBJECTIVE"],
            min_split_gain=0.000001,
            min_child_weight=0.000001,
            min_child_samples=2,
            subsample=1.0,
            subsample_freq=0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=get_seed(task),
            n_jobs=1,
            importance_type="gain",
            **{
                "data_random_seed": None,
                "data_sample_strategy": "goss",
                "extra_seed": None,
                "feature_fraction_seed": None,
                "feature_pre_filter": False,
                "force_col_wise": True,
                "min_data_in_bin": 1,
                "top_rate": 0.5,
                "verbosity": -1,
            },
        )
        # Search space
        space = {
            "estimator__regressor__colsample_bytree": uniform(0.5, 0.5),
            "estimator__regressor__extra_trees": [True, False],
            "estimator__regressor__path_smooth": loguniform(1, 1000),
        }
        # Add scaler to the estimator
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
            func=None,
            inverse_func=None,
            check_inverse=True,
        )
    # Classification
    elif task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass":
        # Estimator
        estimator = LGBMClassifier(
            boosting_type="gbdt",
            num_leaves=100,
            max_depth=-1,
            learning_rate=0.01,
            n_estimators=1000,
            subsample_for_bin=100000,
            objective=task["OBJECTIVE"],
            class_weight=None,
            min_split_gain=0.000001,
            min_child_weight=0.000001,
            min_child_samples=2,
            subsample=1.0,
            subsample_freq=0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=get_seed(task),
            n_jobs=1,
            importance_type="gain",
            **{
                "data_random_seed": None,
                "data_sample_strategy": "goss",
                "extra_seed": None,
                "feature_fraction_seed": None,
                "feature_pre_filter": False,
                "force_col_wise": True,
                "min_data_in_bin": 1,
                "top_rate": 0.5,
                "verbosity": -1,
            },
        )
        # Search space
        space = {
            "estimator__colsample_bytree": uniform(0.5, 0.5),
            "estimator__extra_trees": [True, False],
            "estimator__path_smooth": loguniform(1, 1000),
        }
    # Other
    else:
        # Raise error
        raise ValueError("OBJECTIVE not found.")

    # Make full pipeline ------------------------------------------------------
    # Analyis pipeline
    pipe = Pipeline(
        [("preprocessing", pre_pipe), ("estimator", estimator)],
        memory=None,
        verbose=False,
    ).set_output(transform="pandas")

    # Return pipe and space ---------------------------------------------------
    return pipe, space


def split_data(df, i_trn, i_tst):
    """
    Split dataframe in training and testing dataframes.

    Parameters
    ----------
    df : dataframe
        Dataframe holding the data to split.
    i_trn : numpy array
        Array with indices of training data.
    i_tst : numpy array
        Array with indices of testing data.

    Returns
    -------
    df_trn : dataframe
        Dataframe holding the training data.
    df_tst : dataframe
         Dataframe holding the testing data.
    """

    # Split dataframe via index -----------------------------------------------
    # Dataframe is not empty
    if not df.empty:
        # Make split
        df_trn = df.iloc[i_trn].reset_index(drop=True)
        # Make split
        df_tst = df.iloc[i_tst].reset_index(drop=True)
    # Dataframe is empty
    else:
        # Make empty dataframes
        df_trn, df_tst = pd.DataFrame(), pd.DataFrame()

    # Return train test dataframes --------------------------------------------
    return df_trn, df_tst


def get_class_w(y):
    """
    Compute balanced (inverse-frequency) class weights.

    The weight of class c is ``n_samples / (n_classes * n_c)``, the same
    definition as scikit-learn's ``class_weight='balanced'``. A perfectly
    balanced data set therefore has all weights equal to 1. These weights are
    used (1) as ``class_weight`` of the LightGBM classifier, (2) as sample
    weights of the accuracy in the inner hyperparameter search and in the
    outer scoring (which makes it a balanced accuracy), and (3) to weight the
    confusion matrix plot.

    Parameters
    ----------
    y : array-like or dataframe (n_samples,)
        Class labels.

    Returns
    -------
    class_weights : dict
        Class label -> weight.
    """

    # Count occurances of every class
    counter = Counter(np.asarray(y).ravel().tolist())
    # Number of samples and classes
    n_samples = sum(counter.values())
    n_classes = len(counter)
    # Return inverse frequency weights
    return {key: n_samples / (n_classes * count) for key, count in counter.items()}


def weighted_accuracy_score(y_true, y_pred, class_weights):
    """
    Compute accuracy with every sample weighted by the weight of its class.
    With inverse-frequency weights (see :func:`get_class_w`) this is a
    balanced accuracy.

    Parameters
    ----------
    y_true : ndarray
        True values.
    y_pred : ndarray
        Predicted values.
    class_weights : dictionary
        Class label -> weight, e.g. from :func:`get_class_w`.

    Returns
    -------
    accuracy : float
        Prediction accuracy.
    """

    # Get sample weights ------------------------------------------------------
    # Make sample weights dataframe
    w = y_true.squeeze().map(class_weights).to_numpy()

    # Return sample weighted accuracy -----------------------------------------
    return accuracy_score(y_true, y_pred, sample_weight=w)


def print_tune_summary(task, i_cv, n_splits, hp_params, hp_score):
    """
    Print best paramters and related score to console.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    i_cv : int
        Current cv repetition.
    n_splits : int
        Number of splits in inner cv
    hp_params : dictionary
        Best hyper params found.
    hp_score : dictionary
        Score for best hyper params found.

    Returns
    -------
    None.
    """

    # Print analysis name
    print("Analysis: " + task["ANALYSIS_NAME"])
    # Print data set
    # print('Dataset: '+task['PATH_TO_DATA'])
    # Cross-validation --------------------------------------------------------
    if task["TYPE"] == "CV":
        # Regression
        if task["OBJECTIVE"] == "regression":
            # Print general information
            print(
                str(task["i_y"])
                + "."
                + str(i_cv)
                + " | "
                + "n rep outer cv: "
                + str(task["N_REP_OUTER_CV"])
                + " | "
                + "n rep inner cv: "
                + str(n_splits)
                + " | "
                + "best neg MSE: "
                + str(np.round(hp_score, decimals=4))
            )
        # Classification
        elif task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass":
            # Print general information
            print(
                str(task["i_y"])
                + "."
                + str(i_cv)
                + " | "
                + "n rep outer cv: "
                + str(task["N_REP_OUTER_CV"])
                + " | "
                + "n rep inner cv: "
                + str(n_splits)
                + " | "
                + "acc: "
                + str(np.round(hp_score, decimals=4))
            )
        # Other
        else:
            # Raise error
            raise ValueError("OBJECTIVE not found.")
    # Train-Test split --------------------------------------------------------
    elif task["TYPE"] == "TT":
        # Regression
        if task["OBJECTIVE"] == "regression":
            # Print general information
            print(
                str(task["i_y"])
                + "."
                + str(i_cv)
                + " | "
                + "n rep inner cv: "
                + str(n_splits)
                + " | "
                + "best neg MSE: "
                + str(np.round(hp_score, decimals=4))
            )
        # Classification
        elif task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass":
            # Print general information
            print(
                str(task["i_y"])
                + "."
                + str(i_cv)
                + " | "
                + "n rep inner cv: "
                + str(n_splits)
                + " | "
                + "acc: "
                + str(np.round(hp_score, decimals=4))
            )
        # Other
        else:
            # Raise error
            raise ValueError("OBJECTIVE not found.")
    # Other -------------------------------------------------------------------
    else:
        # Raise error
        raise ValueError("TYPE not found.")
    # Print best hyperparameter and related score for regression task
    print(str(hp_params))

    # Return None -------------------------------------------------------------
    return


def tune_pipe(task, i_cv, pipe, space, g_trn, x_trn, y_trn):
    """
    Inner loop of the nested cross-validation. Runs a search for optimal
    hyperparameter (random search).
    Ref: Hastie T, Tibshirani R, Friedman JH. The elements of statistical
    learning: data mining, inference, and prediction. 2nd ed. New York,
    NY: Springer; 2009.
    Ref: Cawley GC, Talbot NLC. On Over-ﬁtting in Model Selection and
    Subsequent Selection Bias in Performance Evaluation. 2010;(11):2079–107.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    i_cv : int
        Current iteration of outer cross-validation.
    pipe : pipeline object
        Analysis pipeline.
    space : dict
        Space that should be searched for optimale parameters.
    g_trn : ndarray (n_samples)
        Group data.
    x_trn : ndarray (n_features x n_samples)
        Predictor train data.
    y_trn : ndarray (n_samples)
        Target train data.

    Returns
    -------
    pipe : pipeline object
        Fitted pipeline object with tuned parameters.
    best parameters : dict
        Best hyperparameters of the pipe.
    """

    # Get scorer --------------------------------------------------------------
    # Regression
    if task["OBJECTIVE"] == "regression":
        # neg_mean_squared_error
        scorer = "neg_mean_squared_error"
    # Classification
    elif task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass":
        # Weighted accuracy for classification
        scorer = make_scorer(
            weighted_accuracy_score,
            greater_is_better=True,
            **{"class_weights": get_class_w(y_trn)},
        )
        # Add current class weights to the pipe
        pipe.set_params(**{"estimator__class_weight": get_class_w(y_trn)})
    # Other
    else:
        # Raise error
        raise ValueError("OBJECTIVE not found.")

    # Tune analysis pipeline --------------------------------------------------
    # Choose n_repeats to approx N_SAMPLES_INNER_CV predictions, min 1, max 10
    n_repeats = min(10, max(1, int(task["N_SAMPLES_INNER_CV"] / g_trn.shape[0])))
    # Instatiate random parameter search
    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=task["N_SAMPLES_RS"],
        scoring=scorer,
        n_jobs=task["N_JOBS"],
        refit=True,
        cv=RepeatedGroupKFold(
            n_splits=5, n_repeats=n_repeats, random_state=get_seed(task, i_cv)
        ),
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=get_seed(task, i_cv),
        error_score=0,
        return_train_score=False,
    )
    # Random search for best parameter
    search.fit(x_trn, y_trn.squeeze(), groups=g_trn)
    # Print tune summary
    print_tune_summary(task, i_cv, n_repeats, search.best_params_, search.best_score_)

    # Return tuned analysis pipe ----------------------------------------------
    return search.best_estimator_, search.best_params_


def corrected_std(differences, n_tst_over_n_trn=0.25):
    """
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
    """

    # Get corrected std -------------------------------------------------------
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    # Corrected variance
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_tst_over_n_trn)
    # Corrected standard deviation
    corrected_std = np.sqrt(corrected_var)

    # Return corrected standard deviation -------------------------------------
    return corrected_std


def corrected_ttest(differences, n_tst_over_n_trn=0.25):
    """
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
    """

    # Compute t statistics and p value ----------------------------------------
    # Get mean of differences
    mean = np.mean(differences)
    # Get corrected standard deviation
    std = corrected_std(differences, n_tst_over_n_trn)
    # Compute t statistics
    if std >= 0.0 and std <= 0.0000001:  # for technical purposes
        t_stat = 0.0
        p_val = 1.0
    else:
        # Compute p value for right-tailed t-test (H1: mean > 0). A negative
        # t statistic (first model worse than second) is not significant.
        t_stat = mean / std
        p_val = t.sf(t_stat, df=len(differences) - 1)
    # Return t statistics and p value -----------------------------------------
    return t_stat, p_val


REGRESSION_METRICS = ("r2", "mae", "mse", "rmse")
BINARY_METRICS = (
    "acc",
    "auc",
    "logloss",
    "brier",
    "mcc",
    "kappa",
    "f1_macro",
    "mcfadden",
    "cox_snell",
    "nagelkerke",
    "tjur",
)
MULTICLASS_METRICS = tuple(m for m in BINARY_METRICS if m != "tjur")
# Metrics for which smaller is better
LOWER_IS_BETTER = ("mae", "mse", "rmse", "logloss", "brier")


def metrics_for(objective):
    """Names of the scores computed for an objective (first one: main score)."""
    return {
        "regression": REGRESSION_METRICS,
        "binary": BINARY_METRICS,
        "multiclass": MULTICLASS_METRICS,
    }[objective]


def classification_metrics(y_true, y_pred, proba, classes, class_weights):
    """
    Out-of-sample scores of a classifier.

    All probability based metrics are **class-balanced**: every sample is
    weighted by the weight of its class (inverse class frequency, see
    :func:`get_class_w`) and the null model assigns equal probability to
    every class. This matches the balanced class weights the models are
    trained with (their probabilities reflect equal class priors).

    Parameters
    ----------
    y_true, y_pred : array-like (n_samples,)
        True and predicted labels.
    proba : ndarray (n_samples, n_classes)
        Predicted class probabilities, columns ordered as ``classes``.
    classes : array-like (n_classes,)
        Class labels.
    class_weights : dict
        Class label -> weight.

    Returns
    -------
    dict
        ``acc`` balanced accuracy; ``auc`` area under the ROC curve (binary,
        else macro one-vs-rest; not weighted); ``logloss`` weighted log loss;
        ``brier`` weighted Brier score (binary: squared error of the
        positive-class probability, multiclass: summed over classes);
        ``mcc`` Matthews correlation coefficient and ``kappa`` Cohen's kappa
        (both on the labels, unweighted); ``f1_macro`` macro F1;
        ``mcfadden`` ``1 - ll_model / ll_null``; ``cox_snell`` and
        ``nagelkerke`` R2 from the same log-likelihoods (``1 - exp(2 *
        (ll_null - ll_model))``, the latter scaled to a maximum of 1);
        ``tjur`` (binary only, else NaN) mean predicted positive-class
        probability of the positive class minus that of the negative class.
        Here ``ll`` is the weighted mean log-likelihood per sample. The
        pseudo R2 values are computed on test data and can be negative.
    """

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.asarray(classes)
    k = len(classes)
    idx = pd.Index(classes).get_indexer(y_true)
    weights = pd.Series(y_true).map(class_weights).to_numpy(dtype=float)
    out = {m: np.nan for m in BINARY_METRICS}
    out["acc"] = float(np.average(y_pred == y_true, weights=weights))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    out["kappa"] = float(cohen_kappa_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if (idx < 0).any():
        # A test label the model never saw: probability metrics undefined
        return out
    p = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0)
    onehot = np.eye(k)[idx]
    # Log-likelihoods per sample (weighted mean); null: equal probabilities
    ll = float(np.average(np.log(p[np.arange(len(idx)), idx]), weights=weights))
    ll0 = -np.log(k)
    out["logloss"] = -ll
    out["mcfadden"] = 1.0 - ll / ll0
    out["cox_snell"] = 1.0 - np.exp(2.0 * (ll0 - ll))
    out["nagelkerke"] = out["cox_snell"] / (1.0 - np.exp(2.0 * ll0))
    if k == 2:
        out["brier"] = float(np.average((p[:, 1] - onehot[:, 1]) ** 2, weights=weights))
        if (idx == 0).any() and (idx == 1).any():
            out["tjur"] = float(p[idx == 1, 1].mean() - p[idx == 0, 1].mean())
    else:
        out["brier"] = float(
            np.average(((p - onehot) ** 2).sum(axis=1), weights=weights)
        )
    try:
        if k == 2:
            out["auc"] = float(roc_auc_score(onehot[:, 1], p[:, 1]))
        else:
            out["auc"] = float(
                roc_auc_score(
                    y_true,
                    p / p.sum(axis=1, keepdims=True),
                    multi_class="ovr",
                    average="macro",
                    labels=classes,
                )
            )
    except ValueError:
        pass  # a class is missing in the test data
    return {k_: float(v) for k_, v in out.items()}


def score_predictions(task, pipe, x_tst, y_tst, y):
    """
    Compute scores for predictions based on task.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    pipe : pipeline object
        Analysis pipeline.
    x_tst : ndarray (n_features x n_samples)
        Predictor test data.
    y_tst : ndarray (n_samples)
        Target test data.
    y : ndarray
        All available target data to compute true class weights for scoring.

    Returns
    -------
    scores : dict
        Returns scoring results. R², MAE, MSE and RMSE if task is
        regression. If task is classification: predicted probabilities,
        class weights and the metrics of :func:`classification_metrics`.
    """

    # Predict -----------------------------------------------------------------
    # Predict test samples
    y_pred = pipe.predict(x_tst)

    # Score results -----------------------------------------------------------
    # Regression
    if task["OBJECTIVE"] == "regression":
        # Score predictions in terms of mae
        mae = mean_absolute_error(y_tst, y_pred)
        # Score predictions in terms of mse
        mse = mean_squared_error(y_tst, y_pred)
        # Score predictions in terms of R²
        r2 = r2_score(y_tst, y_pred)
        # Results
        scores = {
            "y_true": y_tst.squeeze().to_numpy(),
            "y_pred": y_pred,
            "mae": mae,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": r2,
        }
    # Classification
    elif task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass":
        # Get class weights
        class_weights = get_class_w(y)
        # Predicted probabilities, columns ordered as pipe.classes_
        proba = pipe.predict_proba(x_tst)
        # Metrics (balanced accuracy, AUC, log loss, pseudo R2, ...)
        metrics = classification_metrics(
            y_tst.squeeze().to_numpy(), y_pred, proba, pipe.classes_, class_weights
        )
        if task["OBJECTIVE"] == "multiclass":
            metrics.pop("tjur")
        # Results
        scores = {
            "y_true": y_tst.squeeze().to_numpy(),
            "y_pred": y_pred,
            "y_proba": proba,
            "class_weights": class_weights,
            **metrics,
        }
    # Other
    else:
        # Raise error
        raise ValueError("OBJECTIVE not found.")

    # Return scores -----------------------------------------------------------
    return scores


def aggregate_shap_explanations(explanation, groups, interactions=False):
    """
    Sum the SHAP values of groups of features into one feature each.

    SHAP values are additive, so the contribution of a set of features equals
    the sum of the individual contributions. This is used to fold the
    per-class columns of a multiclass target encoder back into one column per
    predictor. The aggregated feature takes the position of the first member
    of its group.

    Parameters
    ----------
    explanation : shap.Explanation
        Explanation with ``values`` of shape (n_samples, n_features,
        n_outputs) or, if ``interactions`` is True, (n_samples, n_features,
        n_features, n_outputs). Arrays without an outputs axis are handled
        as well.
    groups : dict
        New feature name -> list of feature names to sum.
    interactions : bool, default False
        True if ``values`` holds SHAP interaction values (two feature axes).

    Returns
    -------
    shap.Explanation
        New explanation with aggregated features. ``data`` is carried over
        unchanged and has to be replaced by the caller.
    """

    # Names and group membership ----------------------------------------------
    names = list(explanation.feature_names)
    member_of = {f: g for g, fs in groups.items() for f in fs}
    missing = [f for f in member_of if f not in names]
    if missing:
        raise ValueError(f"Features to aggregate not found: {missing}")
    # New feature names, group placed at position of its first member
    new_names = []
    for f in names:
        g = member_of.get(f, f)
        if g not in new_names:
            new_names.append(g)
    # Aggregation matrix (n_old x n_new)
    agg = np.zeros((len(names), len(new_names)))
    for i, f in enumerate(names):
        agg[i, new_names.index(member_of.get(f, f))] = 1.0

    # Aggregate values --------------------------------------------------------
    v = np.asarray(explanation.values)
    if interactions:
        # Aggregate both feature axes (axes 1 and 2)
        v = np.einsum("ia,jb,nij...->nab...", agg, agg, v)
    else:
        # Aggregate feature axis (axis 1)
        v = np.einsum("ia,ni...->na...", agg, v)

    # Return new explanation --------------------------------------------------
    return Explanation(
        v,
        base_values=explanation.base_values,
        data=explanation.data,
        feature_names=new_names,
        compute_time=explanation.compute_time,
    )


def get_explainations(task, pipe, x_trn, x_tst, max_samples="task"):
    """
    Get SHAP (SHapley Additive exPlainations) model explainations.
    Ref: Molnar, Christoph. 'Interpretable machine learning. A Guide for
    Making Black Box Models Explainable', 2019.
    https://christophm.github.io/interpretable-ml-book/.
    Ref: Lundberg, Scott M., and Su-In Lee. “A unified approach to
    interpreting model predictions.” Advances in Neural Information Processing
    Systems. 2017.
    Ref: Lundberg, Scott M., Gabriel G. Erion, and Su-In Lee. “Consistent
    individualized feature attribution for tree ensembles.” arXiv preprint
    arXiv:1802.03888 (2018).
    Ref: Sundararajan, Mukund, and Amir Najmi. “The many Shapley values for
    model explanation.” arXiv preprint arXiv:1908.08474 (2019).
    Ref: Janzing, Dominik, Lenon Minorics, and Patrick Blöbaum. “Feature
    relevance quantification in explainable AI: A causality problem.” arXiv
    preprint arXiv:1910.13413 (2019).
    Ref: Slack, Dylan, et al. “Fooling lime and shap: Adversarial attacks on
    post hoc explanation methods.” Proceedings of the AAAI/ACM Conference on
    AI, Ethics, and Society. 2020.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    pipe : pipeline object
        Fitted pipeline object with tuned parameters.
    x_trn : ndarray (n_features x n_samples)
        Background data.
    x_tst : dataframe (n_samples x n_features)
        Data for shap computation.
    max_samples : int, None or 'task', default 'task'
        Number of randomly chosen rows of ``x_tst`` to explain. 'task' uses
        ``task['MAX_SAMPLES_SHAP']``, None explains all rows.

    Returns
    -------
    shap_explainations : shap.Explanation
        SHAP values of (a random subsample of) the test rows, in units of the
        original target (regression) or log-odds (classification).
        ``values`` has shape (n, features) for regression, (n, features, 2)
        for binary and (n, features, classes) for multiclass tasks; with
        interaction values (n, features, features) for regression and binary
        and (n, features, features, classes) for multiclass.
        ``data`` holds the unscaled predictor values.
    """

    # Get SHAP test data ------------------------------------------------------
    # Subsample test data
    if max_samples == "task":
        max_samples = task["MAX_SAMPLES_SHAP"]
    if max_samples is None:
        max_samples = x_tst.shape[0]
    x_tst_shap_orig = x_tst.sample(
        n=min(x_tst.shape[0], max_samples),
        random_state=get_seed(task, default=3141592),
        ignore_index=True,
    )
    # Transform shap test data
    x_tst_shap = pipe[0].transform(x_tst_shap_orig)

    # Explainer and Explainations ---------------------------------------------
    # Regression
    if task["OBJECTIVE"] == "regression":
        # Get predictor
        predictor = pipe[1].regressor_
    # Classification
    elif task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass":
        # Get predictor
        predictor = pipe[1]
    # Get explainer
    explainer = TreeExplainer(
        predictor,
        data=None,
        model_output="raw",
        feature_perturbation="tree_path_dependent",
        feature_names=None,
        approximate=False,
    )
    # Get explainations with interactions
    if task["SHAP_INTERACTIONS"]:
        # Get shap values
        shap_explainations = explainer(
            x_tst_shap, interactions=True, check_additivity=False
        )
    # Get explainations without interactions
    elif not task["SHAP_INTERACTIONS"]:
        # Get shap values
        shap_explainations = explainer(
            x_tst_shap, interactions=False, check_additivity=False
        )
    # Other
    else:
        # Raise error
        raise ValueError("Invalid value for SHAP_INTERACTIONS.")

    # Prepare shap_explainations ----------------------------------------------
    # Binary main effects: SHAP returns values for the positive class only.
    # Add a class axis (class 0 = -class 1) so that the figures can treat
    # binary and multiclass tasks alike.
    if (
        task["OBJECTIVE"] == "binary"
        and not task["SHAP_INTERACTIONS"]
        and shap_explainations.values.ndim == 2
    ):
        shap_explainations = Explanation(
            np.stack([-shap_explainations.values, shap_explainations.values], axis=-1),
            base_values=np.stack(
                [
                    -np.asarray(shap_explainations.base_values),
                    np.asarray(shap_explainations.base_values),
                ],
                axis=-1,
            ),
            data=shap_explainations.data,
            feature_names=shap_explainations.feature_names,
            compute_time=shap_explainations.compute_time,
        )
    # Multiclass: a target-encoded predictor is expanded to one column per
    # class. Sum its SHAP values back to one value per predictor.
    if task["OBJECTIVE"] == "multiclass" and task["X_CAT_MULT_NAMES"]:
        # Get fitted target encoder
        te = pipe[0].named_steps["coltrans"].named_transformers_["mult_pred"]
        # Names of the expanded columns of every encoded predictor
        groups = {
            name: [f"{name}_{c}" for c in te.classes_]
            for name in task["X_CAT_MULT_NAMES"]
        }
        # Aggregate
        shap_explainations = aggregate_shap_explanations(
            shap_explainations, groups, interactions=task["SHAP_INTERACTIONS"]
        )
    # Replace scaled data in shap explainations with unscaled
    shap_explainations.data = x_tst_shap_orig[
        shap_explainations.feature_names
    ].to_numpy()
    # If regression
    if task["OBJECTIVE"] == "regression":
        # Rescale shap values from scaled data to original space
        shap_explainations.values = (
            shap_explainations.values * pipe[1].transformer_.scale_[0]
        )
        # Rescale shap base values from scaled data to original space
        shap_explainations.base_values = (
            shap_explainations.base_values * pipe[1].transformer_.scale_[0]
        ) + pipe[1].transformer_.mean_[0]

    # Return shap explainations -----------------------------------------------
    return shap_explainations


def print_current_results(task, t_start, scores, scores_sh):
    """
    Print current results to console.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    t_start : time
        Start time of the current cross-validation loop.
    scores : dict
        Scores dict.
    scores_sh : dict
        Scores with shuffled data dict.

    Returns
    -------
    None.
    """

    # Print results -----------------------------------------------------------
    # Regression
    if task["OBJECTIVE"] == "regression":
        # Print current R2
        print("Current CV loop R2: " + str(np.round(scores[-1]["r2"], decimals=4)))
        # Print running mean R2
        print(
            "Running mean R2: "
            + str(np.round(np.nanmean([i["r2"] for i in scores]), decimals=4))
        )
        # Print running mean shuffle R2
        print(
            "Running shuffle mean R2: "
            + str(np.round(np.nanmean([i["r2"] for i in scores_sh]), decimals=4))
        )
        # Print elapsed time
        print(
            "Elapsed time: " + str(np.round(time() - t_start, decimals=1)), end="\n\n"
        )
    # Classification
    elif task["OBJECTIVE"] == "binary" or task["OBJECTIVE"] == "multiclass":
        # Print current acc
        print("Current CV loop acc: " + str(np.round(scores[-1]["acc"], decimals=4)))
        # Print running mean acc
        print(
            "Running mean acc: "
            + str(np.round(np.nanmean([i["acc"] for i in scores]), decimals=4))
        )
        # Print running mean shuffle acc
        print(
            "Running shuffle mean acc: "
            + str(np.round(np.nanmean([i["acc"] for i in scores_sh]), decimals=4))
        )
        # Print elapsed time
        print(
            "Elapsed time: " + str(np.round(time() - t_start, decimals=1)), end="\n\n"
        )
    # Other
    else:
        # Raise error
        raise ValueError("OBJECTIVE not found.")

    # Return None -------------------------------------------------------------
    return


def _total_compute_time(explainations):
    """Sum of the SHAP compute times (missing times count as 0)."""
    return float(np.sum([k.compute_time or 0 for k in explainations]))


def get_shap_effects(task, explainations, c_class=-1):
    """
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
    """

    # Get shap effects --------------------------------------------------------
    # Case 1: no interaction and regression
    if not task["SHAP_INTERACTIONS"] and task["OBJECTIVE"] == "regression":
        # SHAP effects
        shap_effects = [np.mean(np.abs(k.values), axis=0) for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 2: interaction and regression
    elif task["SHAP_INTERACTIONS"] and task["OBJECTIVE"] == "regression":
        # Get SHAP effects
        shap_effects = [
            np.mean(np.abs(np.sum(k.values, axis=2)), axis=0) for k in explainations
        ]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 3: no interaction and binary
    elif not task["SHAP_INTERACTIONS"] and task["OBJECTIVE"] == "binary":
        # SHAP effects
        shap_effects = [
            np.mean(np.abs(k.values[:, :, c_class]), axis=0) for k in explainations
        ]
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values for k in explainations]))
    # Case 4: interaction and binary
    elif task["SHAP_INTERACTIONS"] and task["OBJECTIVE"] == "binary":
        # Get SHAP effects
        shap_effects = [
            np.mean(np.abs(np.sum(k.values, axis=2)), axis=0) for k in explainations
        ]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 5: no interaction and multiclass
    elif not task["SHAP_INTERACTIONS"] and task["OBJECTIVE"] == "multiclass":
        # SHAP effects
        shap_effects = [
            np.mean(np.abs(k.values[:, :, c_class]), axis=0) for k in explainations
        ]
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values for k in explainations]))
    # Case 6: interaction and multiclass (values: rows x features x features x classes)
    elif task["SHAP_INTERACTIONS"] and task["OBJECTIVE"] == "multiclass":
        # Get SHAP effects of the class
        shap_effects = [
            np.mean(np.abs(np.sum(k.values, axis=2)[:, :, c_class]), axis=0)
            for k in explainations
        ]
        # Base value of the class
        base = np.mean(
            np.hstack([np.asarray(k.base_values)[:, c_class] for k in explainations])
        )
    # Case else
    else:
        # Raise error
        raise ValueError("Unsupported task.")

    # Make SHAP effects dataframe
    shap_effects_df = pd.DataFrame(shap_effects, columns=task["x_names"])

    # Return shap effects -----------------------------------------------------
    return shap_effects_df, base


def get_shap_values(task, explainations, c_class=-1):
    """
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
    """

    # Get shap effects --------------------------------------------------------
    # Case 1: no interaction and regression
    if not task["SHAP_INTERACTIONS"] and task["OBJECTIVE"] == "regression":
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
            compute_time=_total_compute_time(explainations),
        )
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 2: interaction and regression
    elif task["SHAP_INTERACTIONS"] and task["OBJECTIVE"] == "regression":
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
            compute_time=_total_compute_time(explainations),
        )
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 3: no interaction and binary
    elif not task["SHAP_INTERACTIONS"] and task["OBJECTIVE"] == "binary":
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k[:, :, c_class].values for k in explainations]),
            base_values=np.hstack(
                [k[:, :, c_class].base_values for k in explainations]
            ),
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
            compute_time=_total_compute_time(explainations),
        )
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values for k in explainations]))
    # Case 4: interaction and binary
    elif task["SHAP_INTERACTIONS"] and task["OBJECTIVE"] == "binary":
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
            compute_time=_total_compute_time(explainations),
        )
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 5: no interaction and multiclass
    elif not task["SHAP_INTERACTIONS"] and task["OBJECTIVE"] == "multiclass":
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k[:, :, c_class].values for k in explainations]),
            base_values=np.hstack(
                [k[:, :, c_class].base_values for k in explainations]
            ),
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
            compute_time=_total_compute_time(explainations),
        )
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values for k in explainations]))
    # Case 6: interaction and multiclass (values: rows x features x features x classes)
    elif task["SHAP_INTERACTIONS"] and task["OBJECTIVE"] == "multiclass":
        # Explainer object of the class
        shap_explainations = Explanation(
            np.vstack([np.asarray(k.values)[..., c_class] for k in explainations]),
            base_values=np.hstack(
                [np.asarray(k.base_values)[:, c_class] for k in explainations]
            ),
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
            compute_time=_total_compute_time(explainations),
        )
        # Base value of the class
        base = np.mean(
            np.hstack([np.asarray(k.base_values)[:, c_class] for k in explainations])
        )
    # Case else
    else:
        # Raise error
        raise ValueError("Unsupported task.")

    # Return shap effects -----------------------------------------------------
    return shap_explainations, base


CORRECTIONS = ("bonferroni", "holm", "fdr_bh")


def adjust_pvalues(p, method):
    """
    Adjust a family of p-values for multiple comparisons.

    Parameters
    ----------
    p : array-like
        P-values of one family of tests (any shape; NaN values are left
        alone and do not count towards the family size).
    method : {None, 'bonferroni', 'holm', 'fdr_bh'}
        ``None``: no adjustment. ``'bonferroni'``: ``p * m``. ``'holm'``:
        Holm's step-down procedure (controls the family-wise error rate,
        never less powerful than Bonferroni). ``'fdr_bh'``: Benjamini and
        Hochberg's step-up procedure (controls the false discovery rate
        under independence or positive dependence; less conservative).
        Adjusted values are capped at 1.

    Returns
    -------
    ndarray
        Adjusted p-values, same shape as ``p``.
    """
    p = np.asarray(p, dtype=float)
    out = p.copy()
    if method is None:
        return out
    if method not in CORRECTIONS:
        raise ValueError(f"method must be None or one of {CORRECTIONS}, got {method!r}.")
    flat = out.ravel()  # view
    ok = np.isfinite(flat)
    m = int(ok.sum())
    if m == 0:
        return out
    q = flat[ok]
    order = np.argsort(q, kind="stable")
    ranked = q[order]
    if method == "bonferroni":
        adj = np.minimum(q * m, 1.0)
    else:
        if method == "holm":
            step = (m - np.arange(m)) * ranked
            ranked_adj = np.maximum.accumulate(step)
        else:  # fdr_bh
            step = ranked * m / np.arange(1, m + 1)
            ranked_adj = np.minimum.accumulate(step[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.minimum(ranked_adj, 1.0)
    flat[ok] = adj
    return flat.reshape(p.shape)


def effect_pvalues(effects, effects_shuffled):
    """
    One-sided p-values that an effect is larger than that of the shuffled-target
    model, from the corrected resampled t-test over the outer folds.

    Parameters
    ----------
    effects, effects_shuffled : ndarray (n_folds, ...)
        Effects of the real and the shuffled-target model per fold.

    Returns
    -------
    ndarray
        P-values with the shape of the trailing dimensions; NaN where fewer
        than two folds (single train-test split) or values are not finite.
    """
    effects = np.asarray(effects, dtype=float)
    diff = effects - np.asarray(effects_shuffled, dtype=float)
    p = np.full(effects.shape[1:], np.nan)
    if effects.shape[0] < 2:
        return p
    for cell in np.ndindex(*p.shape):
        d = diff[(slice(None),) + cell]
        if np.isfinite(d).all():
            p[cell] = corrected_ttest(d)[1]
    return p


def main_shap_values(explanation, interactions):
    """
    Main-effect SHAP values as array (rows, features, outputs).

    Interaction values are summed over the second feature axis (which gives
    the main SHAP values); values without an outputs axis (regression) get
    one of length 1.
    """
    v = np.asarray(explanation.values)
    if interactions:
        v = v.sum(axis=2)
    if v.ndim == 2:
        v = v[:, :, None]
    return v


def get_interaction_effects(task, explainations, c_class=0):
    """
    Mean absolute SHAP interaction value per outer fold.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    explainations : list of shap explaination objects
        SHAP explainations with interaction values, one per outer fold.
    c_class : int, default 0
        Class to use if the interaction values have a class axis
        (multiclass targets).

    Returns
    -------
    ndarray (n_folds, n_features, n_features)
    """
    arrays = []
    for k in explainations:
        v = np.asarray(k.values)
        if v.ndim == 4:
            v = v[..., c_class]
        arrays.append(np.mean(np.abs(v), axis=0))
    return np.array(arrays)


RESULT_KEYS = (
    "best_params",
    "scores",
    "explainations",
    "scores_sh",
    "explainations_sh",
)


def _append_fold(results, fold):
    """Append the entries of one outer fold to the results lists."""
    for key in RESULT_KEYS:
        results[key].append(fold[key])


def _run_fold(task, i_fold, pipe, space, g_trn, x_trn, y_trn, x_tst, y_tst, y):
    """
    Tune, score and explain one outer fold, for the real and the shuffled
    target model.

    Returns
    -------
    pipe : pipeline
        Fitted pipeline with tuned parameters.
    fold : dict
        Keys ``RESULT_KEYS``, one entry each.
    """

    # Tune and fit ------------------------------------------------------------
    pipe, params = tune_pipe(task, i_fold, pipe, space, g_trn, x_trn, y_trn)
    # Analyze -----------------------------------------------------------------
    scores = score_predictions(task, pipe, x_tst, y_tst, y)
    explainations = get_explainations(task, pipe, x_trn, x_tst)
    # Shuffle data analyze ----------------------------------------------------
    # Clone pipe (keeps the tuned hyperparameters, refits from scratch)
    pipe_sh = clone(pipe)
    # Refit pipe with shuffled targets
    pipe_sh.fit(
        x_trn,
        shuffle(
            y_trn, random_state=get_seed(task, i_fold, default=DEFAULT_SEED)
        ).squeeze(),
    )
    scores_sh = score_predictions(task, pipe_sh, x_tst, y_tst, y)
    explainations_sh = get_explainations(task, pipe_sh, x_trn, x_tst)
    return pipe, {
        "best_params": params,
        "scores": scores,
        "explainations": explainations,
        "scores_sh": scores_sh,
        "explainations_sh": explainations_sh,
    }


def cross_validation(task, g, x, y, on_fold=None, done=None):
    """
    Perform the repeated group 5-fold cross-validation (``evaluation='CV'``).
    The outer split uses ``task['N_REP_OUTER_CV']`` repetitions; the number
    of folds (5) is fixed. Each outer fold: tune on the training part, score
    and explain on the test part, then repeat with shuffled training targets.
    Ref: Hastie T, Tibshirani R, Friedman JH. The elements of statistical
    learning: data mining, inference, and prediction. 2nd ed. New York,
    NY: Springer; 2009
    Ref: Cawley GC, Talbot NLC. On Over-fitting in Model Selection and
    Subsequent Selection Bias in Performance Evaluation. 2010;(11):2079-107.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    g : dataframe
        Groups dataframe.
    x : dataframe
        Predictors dataframe.
    y : dataframe
        Target dataframe.
    on_fold : callable or None, default None
        Called as ``on_fold(i_fold, fold)`` after every newly computed fold
        (used to store results incrementally).
    done : list of dict or None, default None
        Already computed folds (in order, from fold 0) to reuse instead of
        recomputing them (resume).

    Returns
    -------
    results : dict
        Keys ``RESULT_KEYS``; every value is a list with one entry per fold.
    """

    # Initialize --------------------------------------------------------------
    results = {key: [] for key in RESULT_KEYS}
    done = [] if done is None else done
    # Get analysis pipeline and space
    pipe, space = prepare(task)

    # Main cross-validation loop ----------------------------------------------
    # Instatiate main cv splitter with fixed random state for comparison
    cv = RepeatedGroupKFold(
        n_splits=5,
        n_repeats=task["N_REP_OUTER_CV"],
        random_state=get_seed(task, default=DEFAULT_SEED),
    )
    # Loop over main (outer) cross validation splits
    for i_cv, (i_trn, i_tst) in tqdm(enumerate(cv.split(g, groups=g))):
        # Reuse stored fold (resume)
        if i_cv < len(done):
            _append_fold(results, done[i_cv])
            continue
        # Save loop start time
        t_start = time()
        # Split data
        g_trn, g_tst = split_data(g, i_trn, i_tst)
        y_trn, y_tst = split_data(y, i_trn, i_tst)
        x_trn, x_tst = split_data(x, i_trn, i_tst)
        # Tune, score, explain
        pipe, fold = _run_fold(
            task, i_cv, pipe, space, g_trn, x_trn, y_trn, x_tst, y_tst, y
        )
        _append_fold(results, fold)
        if on_fold is not None:
            on_fold(i_cv, fold)
        # Print current results
        print_current_results(task, t_start, results["scores"], results["scores_sh"])

    # Return results ----------------------------------------------------------
    return results


def get_test_groups(g, test_size, seed):
    """
    Randomly choose whole groups as test set for a train-test split.

    Groups are shuffled and added to the test set until it holds at least
    ``test_size`` of all rows. If every row is its own group (no grouping
    variable) this is a plain random subset of rows without replacement.

    Parameters
    ----------
    g : dataframe (n_samples, 1)
        Group labels.
    test_size : float
        Wanted fraction of rows in the test set, 0 < test_size < 1.
    seed : int or None
        Seed of the random number generator.

    Returns
    -------
    i_tst : ndarray of int
        Sorted positions (not labels) of the test rows in ``g``.
    """

    # Get group labels and unique groups
    labels = np.asarray(g).ravel()
    unique_groups = pd.unique(labels)
    # Shuffle groups
    rng = np.random.default_rng(seed)
    unique_groups = unique_groups[rng.permutation(len(unique_groups))]
    # Number of wanted test rows (at least one, at most all but one group)
    n_wanted = max(1, int(np.ceil(test_size * len(labels))))
    # Add groups until enough rows
    counts = pd.Series(labels).value_counts()
    chosen, n_rows = [], 0
    for group in unique_groups[:-1]:
        if n_rows >= n_wanted:
            break
        chosen.append(group)
        n_rows += counts[group]
    # Return sorted positions of test rows
    return np.flatnonzero(np.isin(labels, chosen))


def train_test_split(task, g, x, y, i_tst, on_fold=None, done=None):
    """
    Perform a single train-test split analysis (``evaluation='TT'``).
    Ref: Hastie T, Tibshirani R, Friedman JH. The elements of statistical
    learning: data mining, inference, and prediction. 2nd ed. New York,
    NY: Springer; 2009
    Ref: Cawley GC, Talbot NLC. On Over-fitting in Model Selection and
    Subsequent Selection Bias in Performance Evaluation. 2010;(11):2079-107.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    g : dataframe
        Groups dataframe.
    x : dataframe
        Predictors dataframe.
    y : dataframe
        Target dataframe.
    i_tst : ndarray of int
        Positions (not labels) of the test rows in ``g``, ``x`` and ``y``.
        All other rows are used for training.
    on_fold, done :
        As in :func:`cross_validation`; ``done`` holds at most one fold.

    Returns
    -------
    results : dict
        Keys ``RESULT_KEYS``; every value is a list with one entry.
    """

    results = {key: [] for key in RESULT_KEYS}
    if done:
        _append_fold(results, done[0])
        return results
    pipe, space = prepare(task)
    t_start = time()
    # Split data
    i_trn = np.setdiff1d(np.arange(len(g)), i_tst)
    g_trn, g_tst = split_data(g, i_trn, i_tst)
    y_trn, y_tst = split_data(y, i_trn, i_tst)
    x_trn, x_tst = split_data(x, i_trn, i_tst)
    # Tune, score, explain
    pipe, fold = _run_fold(task, 0, pipe, space, g_trn, x_trn, y_trn, x_tst, y_tst, y)
    _append_fold(results, fold)
    if on_fold is not None:
        on_fold(0, fold)
    print_current_results(task, t_start, results["scores"], results["scores_sh"])
    return results


def fit_final(task, g, x, y):
    """
    Tune and fit one pipeline on all rows (used for ``refit=True``).

    The hyperparameters are searched with the same inner cross-validation as
    in the outer folds, but on all data.

    Returns
    -------
    pipe : pipeline
        Fitted pipeline.
    params : dict
        Selected hyperparameters.
    """

    pipe, space = prepare(task)
    # A seed offset that is never used by an outer fold
    return tune_pipe(task, 100000, pipe, space, g, x, y)
