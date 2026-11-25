"""
Scikit-learn style estimators: GBDT with nested cross-validation and SHAP.

* :class:`NCVGBDTRegressor` and :class:`NCVGBDTClassifier`
* ``fit(X, y, groups=None)`` runs the nested cross-validation (evaluation) and
  optionally refits one final model on all data (``refit=True``).
* Results are fitted attributes (``cv_results_``, ``shap_effects_``, ...).
* ``output_dir`` stores everything as JSON while fitting (and allows to resume
  an interrupted run); :func:`load_model` restores a fitted estimator.

Layout of ``output_dir``::

    manifest.json          settings, library versions, data fingerprint, ...
    folds/fold_0000.json   one file per outer fold (scores, parameters, SHAP)
    final_model.joblib     only with refit=True (joblib/pickle, see load_model)
"""

import hashlib
import importlib.metadata
import platform
import shutil
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from . import _json
from .fitting import DEFAULT_SEED
from .fitting import cross_validation
from .fitting import fit_final
from .fitting import get_explainations
from .fitting import get_te_target_type
from .fitting import get_test_groups
from .fitting import train_test_split

FORMAT_VERSION = 1
MANIFEST = "manifest.json"
FOLDS_DIR = "folds"
FINAL_MODEL = "final_model.joblib"
LIBRARIES = ("numpy", "pandas", "scikit-learn", "lightgbm", "shap", "scipy", "imltools")


def _versions():
    """Versions of Python and the libraries the results depend on."""
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _as_list(value, name):
    """Return ``value`` as list of str (None -> empty list)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    value = list(value)
    if not all(isinstance(v, str) for v in value):
        raise TypeError(f"{name} must be a string or a list of strings.")
    return value


def _source_column(X, obj):
    """
    Name of the column of ``X`` that ``obj`` stands for: ``obj`` itself if it
    is a string, the name of a Series that equals that column of ``X``,
    otherwise None.
    """
    if isinstance(obj, str):
        return obj
    name = getattr(obj, "name", None)
    if isinstance(obj, pd.Series) and name in X.columns:
        if X[name].reset_index(drop=True).equals(obj.reset_index(drop=True)):
            return name
    return None


def _fingerprint(x, y, g):
    """Hash of the data used for fitting (to recognize a resumable run)."""
    h = hashlib.sha256()
    for part in (x, y.to_frame(), g):
        h.update(pd.util.hash_pandas_object(part, index=False).to_numpy().tobytes())
        h.update(",".join(map(str, part.columns)).encode())
    return h.hexdigest()


class _NCVGBDTBase(BaseEstimator):
    """
    Common implementation of the regressor and the classifier.
    Use :class:`NCVGBDTRegressor` or :class:`NCVGBDTClassifier`.
    """

    def __init__(
        self,
        x_con_names=None,
        x_cat_bin_names=None,
        x_cat_mult_names=None,
        *,
        evaluation="CV",
        n_jobs=-2,
        n_rep_outer_cv=10,
        n_samples_inner_cv=10000,
        n_samples_rs=100,
        max_samples_shap=100,
        shap_interactions=False,
        test_size=0.2,
        refit=False,
        seed=DEFAULT_SEED,
        analysis_name="analysis",
        output_dir=None,
        resume=False,
    ):
        self.x_con_names = x_con_names
        self.x_cat_bin_names = x_cat_bin_names
        self.x_cat_mult_names = x_cat_mult_names
        self.evaluation = evaluation
        self.n_jobs = n_jobs
        self.n_rep_outer_cv = n_rep_outer_cv
        self.n_samples_inner_cv = n_samples_inner_cv
        self.n_samples_rs = n_samples_rs
        self.max_samples_shap = max_samples_shap
        self.shap_interactions = shap_interactions
        self.test_size = test_size
        self.refit = refit
        self.seed = seed
        self.analysis_name = analysis_name
        self.output_dir = output_dir
        self.resume = resume

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, groups=None, test_idx=None):
        """
        Run the nested cross-validation (and the final refit if requested).

        Rows with a missing value in ``y``, ``groups`` or any used predictor
        are dropped (complete-case analysis); ``used_index_`` tells which
        rows were used.

        Parameters
        ----------
        X : pandas.DataFrame (n_samples, n_columns)
            Predictors. Only the columns named in ``x_con_names``,
            ``x_cat_bin_names`` and ``x_cat_mult_names`` are used. If none
            of these is given, all columns except the target and group
            columns are used as continuous.
        y : str or array-like (n_samples,)
            Target: the name of a column of ``X``, or an array. A pandas
            Series' name (or the column name) is used as target name in file
            and figure names (default 'y'). Regression: numeric.
            Classification: any labels. A target that is a column of ``X``
            is never used as predictor by default; listing it explicitly as
            predictor raises an error.
        groups : str, array-like (n_samples,) or None, default None
            Group of every row (e.g. participant): the name of a column of
            ``X``, or an array. Groups are never split across train and
            test sets. None: every row is its own group.
        test_idx : array-like of int or None, default None
            ``evaluation='TT'`` only: positions (0-based) in ``X`` of the
            test rows. None: random whole groups, see ``test_size``.

        Returns
        -------
        self
        """

        # Check and normalize input -------------------------------------------
        evaluation = str(self.evaluation).upper()
        if evaluation not in ("CV", "TT"):
            raise ValueError(
                f"evaluation must be 'CV' or 'TT', got {self.evaluation!r}."
            )
        if not isinstance(self.shap_interactions, (bool, np.bool_)):
            raise ValueError("shap_interactions can be True or False only.")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if self.resume and self.output_dir is None:
            raise ValueError("resume=True needs output_dir.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        n = len(X)
        # Target and groups given as column names of X, or as arrays that are
        # columns of X: remember which columns they are
        y_col = _source_column(X, y)
        g_col = None if groups is None else _source_column(X, groups)
        for label, col, value in (("y", y_col, y), ("groups", g_col, groups)):
            if isinstance(value, str) and col not in X.columns:
                raise KeyError(f"Column {col!r} given as {label} is not in X.")
        if isinstance(y, str):
            y = X[y]
        if isinstance(groups, str):
            groups = X[groups]
        x_con = _as_list(self.x_con_names, "x_con_names")
        x_bin = _as_list(self.x_cat_bin_names, "x_cat_bin_names")
        x_mult = _as_list(self.x_cat_mult_names, "x_cat_mult_names")
        if not (x_con or x_bin or x_mult):
            # All columns except the target and the group column
            x_con = [c for c in X.columns if c not in (y_col, g_col)]
        x_names = x_con + x_bin + x_mult
        if y_col is not None and y_col in x_names:
            raise ValueError(
                f"The target {y_col!r} is also used as a predictor (leak)."
            )
        if len(set(x_names)) != len(x_names):
            raise ValueError("A predictor is listed more than once.")
        missing = [c for c in x_names if c not in X.columns]
        if missing:
            raise KeyError(f"Columns not in X: {missing}")
        # Target
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "y must have one column; fit one estimator per target."
                )
            y = y.iloc[:, 0]
        name = getattr(y, "name", None)
        name = str(name) if name is not None else "y"
        y = pd.Series(np.asarray(y).ravel(), name=name)
        if len(y) != n:
            raise ValueError("X and y have different lengths.")
        # Groups
        if groups is None:
            g = pd.DataFrame({"idx": np.arange(n)})
        else:
            groups = np.asarray(groups).ravel()
            if len(groups) != n:
                raise ValueError("X and groups have different lengths.")
            g = pd.DataFrame({"groups": groups})

        # Complete cases --------------------------------------------------------
        x = X[x_names].reset_index(drop=True)
        mask = (x.notna().all(axis=1) & y.notna() & g.notna().all(axis=1)).to_numpy()
        pos = np.arange(n)
        print(f"DV={name}, N={int(mask.sum())} of {n}")
        if mask.sum() < 10:
            raise ValueError("Fewer than 10 complete rows.")
        # Categories of target encoded predictors (does not use targets)
        te_categories = [np.sort(x[c].dropna().unique()) for c in x_mult]

        # Objective ---------------------------------------------------------------
        objective, classes = self._get_objective(y[mask])
        if objective == "multiclass" and self.shap_interactions:
            raise ValueError(
                "shap_interactions=True is not supported for multiclass "
                "targets (per-class interaction figures are not "
                "implemented). Use shap_interactions=False."
            )
        get_te_target_type(objective)

        # Task description ----------------------------------------------------
        analysis_name = f"{self.analysis_name}_{evaluation}"
        if self.shap_interactions:
            analysis_name += "_inter"
        task = {
            "TYPE": evaluation,
            "N_JOBS": self.n_jobs,
            "SEED": self.seed,
            "N_REP_OUTER_CV": self.n_rep_outer_cv,
            "N_SAMPLES_INNER_CV": self.n_samples_inner_cv,
            "N_SAMPLES_RS": self.n_samples_rs,
            "MAX_SAMPLES_SHAP": self.max_samples_shap,
            "SHAP_INTERACTIONS": bool(self.shap_interactions),
            "ANALYSIS_NAME": analysis_name,
            "OBJECTIVE": objective,
            "G_NAME": "idx" if groups is None else "groups",
            "X_CON_NAMES": x_con,
            "X_CAT_BIN_NAMES": x_bin,
            "X_CAT_MULT_NAMES": x_mult,
            "Y_NAMES": [name],
            "x_names": x_names,
            "te_categories": te_categories,
            "i_y": 0,
            "y_name": [name],
        }
        if classes is not None:
            task["n_classes"] = len(classes)

        # Filtered data -----------------------------------------------------------
        x_f = x[mask].reset_index(drop=True)
        y_f = y[mask].reset_index(drop=True).to_frame(name)
        g_f = g[mask].reset_index(drop=True)

        # Test set (TT) -----------------------------------------------------------
        i_tst = None
        if evaluation == "TT":
            if test_idx is not None:
                i_tst = np.flatnonzero(np.isin(pos[mask], np.asarray(test_idx)))
            else:
                i_tst = get_test_groups(g_f, self.test_size, self.seed)
            if not 0 < len(i_tst) < len(g_f):
                raise ValueError("Test set is empty or holds all rows.")
            g_all = g_f.to_numpy().ravel()
            shared = np.intersect1d(g_all[i_tst], np.delete(g_all, i_tst))
            if len(shared) and groups is not None:
                warnings.warn(
                    f"{len(shared)} groups are in both train and "
                    "test set (information leak)."
                )
            task["TEST_SET_IND_USED"] = pos[mask][i_tst].tolist()

        # Output directory, resume ------------------------------------------------
        fingerprint = _fingerprint(x_f, y_f.iloc[:, 0], g_f)
        out = None if self.output_dir is None else Path(self.output_dir)
        done = []
        self.task_ = task
        self.feature_names_in_ = np.array(x_names, dtype=object)
        self.n_features_in_ = len(x_names)
        self.target_name_ = name
        self.objective_ = objective
        self.used_index_ = pos[mask]
        self.n_samples_used_ = int(mask.sum())
        if classes is not None:
            self.classes_ = classes
        self._fingerprint = fingerprint
        for attr in ("results_", "best_estimator_", "best_params_"):
            if hasattr(self, attr):
                delattr(self, attr)
        if out is not None:
            done = self._prepare_output(out)

        # Fit ---------------------------------------------------------------------
        on_fold = (
            None if out is None else (lambda i, fold: self._write_fold(out, i, fold))
        )
        if evaluation == "CV":
            results = cross_validation(task, g_f, x_f, y_f, on_fold=on_fold, done=done)
        else:
            results = train_test_split(
                task, g_f, x_f, y_f, i_tst, on_fold=on_fold, done=done
            )
        self.results_ = results

        # Final model -------------------------------------------------------------
        if self.refit:
            final_path = None if out is None else out / FINAL_MODEL
            if self.resume and final_path is not None and final_path.exists():
                self.best_estimator_ = joblib.load(final_path)
                self.best_params_ = self.best_estimator_.get_params()
            else:
                pipe, params = fit_final(task, g_f, x_f, y_f)
                self.best_estimator_, self.best_params_ = pipe, params
                if final_path is not None:
                    joblib.dump(pipe, final_path)
        if out is not None:
            self._write_manifest(out, complete=True)
        return self

    def _get_objective(self, y):
        raise NotImplementedError

    # ------------------------------------------------------------- storage
    def _params_for_manifest(self):
        """Constructor parameters in a JSON-able form."""
        params = self.get_params()
        if params["output_dir"] is not None:
            params["output_dir"] = str(params["output_dir"])
        return params

    def _write_manifest(self, out, complete):
        manifest = {
            "format_version": FORMAT_VERSION,
            "estimator": type(self).__name__,
            "params": self._params_for_manifest(),
            "task": self.task_,
            "versions": _versions(),
            "fingerprint": self._fingerprint,
            "complete": bool(complete),
            "n_folds": len(list((out / FOLDS_DIR).glob("fold_*.json"))),
            "final_model": (out / FINAL_MODEL).exists(),
            "feature_names_in": list(self.feature_names_in_),
            "target_name": self.target_name_,
            "objective": self.objective_,
            "classes": getattr(self, "classes_", None),
            "used_index": self.used_index_,
            "n_samples_used": self.n_samples_used_,
        }
        _json.dump(manifest, out / MANIFEST)

    def _write_fold(self, out, i, fold):
        _json.dump({"index": i, **fold}, out / FOLDS_DIR / f"fold_{i:04d}.json")
        self._write_manifest(out, complete=False)

    def _prepare_output(self, out):
        """Create ``out``; return stored folds to reuse (resume) or clear."""
        out.mkdir(parents=True, exist_ok=True)
        manifest_path = out / MANIFEST
        done = []
        if self.resume and manifest_path.exists():
            old = _json.load(manifest_path)
            same = old.get("fingerprint") == self._fingerprint and _comparable(
                old.get("params")
            ) == _comparable(_json.decode(_json.encode(self._params_for_manifest())))
            if not same:
                raise ValueError(
                    f"{out} holds results from different data or settings. "
                    "Use another output_dir or resume=False to overwrite."
                )
            folds_dir = out / FOLDS_DIR
            files = sorted(folds_dir.glob("fold_*.json"))
            for i, f in enumerate(files):
                fold = _json.load(f)
                if fold["index"] != i:
                    break
                done.append({k: fold[k] for k in fold if k != "index"})
            return done
        # Fresh start: remove files of an earlier run (only our own files)
        shutil.rmtree(out / FOLDS_DIR, ignore_errors=True)
        for name in (MANIFEST, FINAL_MODEL):
            (out / name).unlink(missing_ok=True)
        (out / FOLDS_DIR).mkdir()
        self._write_manifest(out, complete=False)
        return done

    def save(self, path=None):
        """
        Write the fitted estimator to ``path`` (default: ``output_dir``).

        Not needed if ``output_dir`` was set before ``fit``: the files are
        written while fitting.
        """
        self._check_fitted()
        if path is None and self.output_dir is None:
            raise ValueError("Give a path or set output_dir.")
        out = Path(self.output_dir if path is None else path)
        out.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(out / FOLDS_DIR, ignore_errors=True)
        (out / FOLDS_DIR).mkdir()
        n = len(self.results_["scores"])
        for i in range(n):
            fold = {
                k: self.results_[k][i]
                for k in (
                    "best_params",
                    "scores",
                    "explainations",
                    "scores_sh",
                    "explainations_sh",
                )
            }
            _json.dump({"index": i, **fold}, out / FOLDS_DIR / f"fold_{i:04d}.json")
        if hasattr(self, "best_estimator_"):
            joblib.dump(self.best_estimator_, out / FINAL_MODEL)
        self._write_manifest(out, complete=True)
        return out

    @classmethod
    def load(cls, path):
        """
        Restore a fitted estimator from a directory written by ``fit`` with
        ``output_dir`` or by :meth:`save`. See :func:`load_model`.
        """
        return load_model(path, _expected=cls)

    # ------------------------------------------------------------ attributes
    def _check_fitted(self):
        if not hasattr(self, "results_"):
            raise NotFittedError("This estimator is not fitted yet.")

    @property
    def cv_results_(self):
        """
        DataFrame, one row per outer fold: ``score`` (R2 or balanced
        accuracy) and ``score_shuffled`` (same for the shuffled-target
        model), all metrics (``r2``, ``mae``, ``mse`` / ``acc``, each also
        ``*_shuffled``), and the tuned hyperparameters (``param_*``).
        """
        self._check_fitted()
        r = self.results_
        main = "r2" if self.objective_ == "regression" else "acc"
        metrics = ("r2", "mae", "mse") if main == "r2" else ("acc",)
        rows = []
        for i, (bp, sc, ss) in enumerate(
            zip(r["best_params"], r["scores"], r["scores_sh"])
        ):
            row = {"fold": i, "score": sc[main], "score_shuffled": ss[main]}
            for m in metrics:
                row[m] = sc[m]
                row[m + "_shuffled"] = ss[m]
            for k, v in bp.items():
                row["param_" + k.split("__")[-1]] = v
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def score_(self):
        """Mean outer-fold score (R2 or balanced accuracy), NaN ignored."""
        return float(np.nanmean(self.cv_results_["score"]))

    @property
    def score_shuffled_(self):
        """Mean outer-fold score of the shuffled-target baseline."""
        return float(np.nanmean(self.cv_results_["score_shuffled"]))

    @property
    def explanations_(self):
        """List (one per outer fold) of ``shap.Explanation`` objects."""
        self._check_fitted()
        return self.results_["explainations"]

    @property
    def explanations_shuffled_(self):
        """Same for the shuffled-target models."""
        self._check_fitted()
        return self.results_["explainations_sh"]

    @property
    def shap_effects_(self):
        """
        Long DataFrame of SHAP effects (mean absolute SHAP value over the
        explained test rows) with columns ``fold``, ``class`` (None for
        regression; for binary tasks the positive class only), ``feature``,
        ``effect`` and ``effect_shuffled`` (shuffled-target model).
        With interaction values the effect is that of the summed main and
        interaction contributions of a feature.
        """
        self._check_fitted()
        classes = getattr(self, "classes_", None)
        rows = []
        for i, (e, es) in enumerate(
            zip(self.explanations_, self.explanations_shuffled_)
        ):
            a = self._effects(e)
            b = self._effects(es)
            for c in range(a.shape[1]):
                label = (
                    None
                    if classes is None
                    else (classes[1] if self.objective_ == "binary" else classes[c])
                )
                for j, f in enumerate(e.feature_names):
                    rows.append(
                        {
                            "fold": i,
                            "class": label,
                            "feature": f,
                            "effect": a[j, c],
                            "effect_shuffled": b[j, c],
                        }
                    )
        return pd.DataFrame(rows)

    def _effects(self, e):
        """Mean |SHAP| as array (features, outputs)."""
        v = np.asarray(e.values)
        if self.task_["SHAP_INTERACTIONS"]:
            v = v.sum(axis=2)
        elif self.objective_ == "binary":
            v = v[:, :, 1]
        if v.ndim == 2:
            v = v[:, :, None]
        return np.abs(v).mean(axis=0)

    # ------------------------------------------------- final model (refit)
    def _require_final(self):
        if not hasattr(self, "best_estimator_"):
            raise NotFittedError(
                "No final model. Fit with refit=True (or load a model that "
                "was fitted with refit=True)."
            )

    def predict(self, X):
        """
        Predict with the final model (needs ``refit=True``).

        Parameters
        ----------
        X : pandas.DataFrame
            Must hold the columns in ``feature_names_in_``.
        """
        self._require_final()
        return self.best_estimator_.predict(X[list(self.feature_names_in_)])

    def explain(self, X, max_samples=None):
        """
        SHAP explanations of the final model (needs ``refit=True``).

        Parameters
        ----------
        X : pandas.DataFrame
            Rows to explain; must hold the columns in ``feature_names_in_``.
        max_samples : int or None, default None
            Explain at most this many randomly chosen rows; None: all rows.

        Returns
        -------
        shap.Explanation
            Same layout as the entries of ``explanations_``.
        """
        self._require_final()
        x = X[list(self.feature_names_in_)].reset_index(drop=True)
        return get_explainations(
            self.task_, self.best_estimator_, None, x, max_samples=max_samples
        )


class NCVGBDTRegressor(_NCVGBDTBase):
    """
    Gradient boosted trees (LightGBM) for a continuous target, evaluated
    with nested cross-validation and explained with SHAP.

    Parameters
    ----------
    x_con_names : list of str or None, default None
        Continuous predictors (standardized). If none of the three predictor
        lists is given, every column of ``X`` is used as continuous.
    x_cat_bin_names : list of str or None, default None
        Binary categorical predictors; must be numeric (code them 0/1).
    x_cat_mult_names : list of str or None, default None
        Categorical predictors with more than two levels. Target encoded
        (scikit-learn ``TargetEncoder``) inside the pipeline, i.e. fitted on
        training data only.
    evaluation : {'CV', 'TT'}, default 'CV'
        'CV': repeated group 5-fold cross-validation as outer loop (case
        insensitive). 'TT': a single group-disjoint train-test split, see
        ``test_size`` and ``fit(test_idx=...)``.
    n_jobs : int, default -2
        Parallel jobs of the hyperparameter search (-1: all cores, -2: all
        but one, as in scikit-learn).
    n_rep_outer_cv : int, default 10
        'CV' only: repetitions of the outer 5-fold CV with different group
        assignments.
    n_samples_inner_cv : int, default 10000
        Aim for this many predictions in the inner 5-fold CV; the inner CV
        is repeated ``floor(n_samples_inner_cv / n_train)`` times (at least
        1, at most 10).
    n_samples_rs : int, default 100
        Random hyperparameter settings tried per inner search.
    max_samples_shap : int, default 100
        SHAP values are computed for at most this many randomly chosen rows
        of every outer test set.
    shap_interactions : bool, default True
        Also compute SHAP interaction values (slow, memory: features
        squared). Not supported for multiclass targets.
    test_size : float, default 0.2
        'TT' only: wanted fraction of test rows when ``test_idx`` is not
        given; whole groups are drawn at random.
    refit : bool, default False
        Also tune and fit one final model on all rows (one more inner
        search). Required for ``predict``, ``explain`` and
        ``best_estimator_``. The performance estimate always comes from the
        outer folds, never from this model.
    seed : int or None, default 3141592
        Seed for every random step. The same seed gives identical results
        on the same software versions. None: the outer split and SHAP
        subsample use 3141592, everything else is unseeded.
    analysis_name : str, default 'analysis'
        Label used in figure titles and file names ('_<evaluation>', and
        '_inter' for interactions, is appended).
    output_dir : str, path or None, default None
        Directory to which results are written as JSON while fitting (one
        file per outer fold). None: keep results in memory only (use
        ``save`` later). Use one directory per estimator.
    resume : bool, default False
        Continue an interrupted fit in ``output_dir``: finished folds are
        loaded instead of recomputed. The data and all settings must be
        unchanged, otherwise an error is raised.

    Attributes
    ----------
    cv_results_ : DataFrame
        Per outer fold: scores, shuffled-target scores, tuned parameters.
    score_, score_shuffled_ : float
        Mean outer-fold score of the model and of the shuffled baseline.
    shap_effects_ : DataFrame
        Mean absolute SHAP value per fold and feature (long format).
    explanations_, explanations_shuffled_ : list of shap.Explanation
        One entry per outer fold.
    feature_names_in_ : ndarray of str
        Predictors in the order used.
    n_features_in_, n_samples_used_ : int
    used_index_ : ndarray of int
        Positions in ``X`` of the complete-case rows that were used.
    target_name_ : str
    task_, results_ : dict
        Settings and raw results (as used by the figures).
    best_estimator_, best_params_ :
        Final fitted pipeline and its parameters (only with ``refit=True``).

    Examples
    --------
    >>> model = NCVGBDTRegressor(x_con_names=['wt'], output_dir='out')
    >>> model.fit(df[['wt']], df['mpg'])               # doctest: +SKIP
    >>> model.cv_results_                              # doctest: +SKIP
    >>> make_figures(model, figures_dir='out/figures')  # doctest: +SKIP
    """

    def _get_objective(self, y):
        if not pd.api.types.is_numeric_dtype(y):
            raise TypeError("The target of a regressor must be numeric.")
        return "regression", None


class NCVGBDTClassifier(_NCVGBDTBase):
    """
    Gradient boosted trees (LightGBM) for a categorical target, evaluated
    with nested cross-validation and explained with SHAP.

    All parameters, attributes and methods are those of
    :class:`NCVGBDTRegressor`, with these differences: the target may have
    any labels; two classes are fitted as binary, three or more as
    multiclass (``classes_`` holds the labels in sorted order). Performance
    is the **balanced accuracy** (inverse-frequency class weights).
    ``shap_interactions=True`` is not supported for more than two classes.
    Additional method: ``predict_proba`` (needs ``refit=True``).
    """

    def _get_objective(self, y):
        classes = np.unique(y.to_numpy())
        if len(classes) < 2:
            raise ValueError("The target has fewer than two classes.")
        return ("binary" if len(classes) == 2 else "multiclass"), classes

    def predict_proba(self, X):
        """
        Class probabilities of the final model (needs ``refit=True``),
        columns ordered as ``classes_``.
        """
        self._require_final()
        return self.best_estimator_.predict_proba(X[list(self.feature_names_in_)])


def _comparable(params):
    """Parameters that decide whether stored folds can be reused."""
    if params is None:
        return None
    return {
        k: v for k, v in params.items() if k not in ("output_dir", "resume", "refit")
    }


def load_model(path, _expected=None):
    """
    Restore a fitted estimator from a directory.

    Parameters
    ----------
    path : str or path
        Directory written by ``fit`` (with ``output_dir``) or ``save``.

    Returns
    -------
    NCVGBDTRegressor or NCVGBDTClassifier
        With all fitted attributes. If the run was interrupted, only the
        finished folds are loaded (a warning is issued); continue it with
        ``resume=True``.

    Notes
    -----
    All results are plain JSON. Only the final model of ``refit=True``
    (``final_model.joblib``) is a pickle-based file: load directories you
    trust only, and expect it to fail on different scikit-learn / LightGBM
    versions. A warning is issued if library versions differ from those the
    results were created with.
    """
    path = Path(path)
    manifest = _json.load(path / MANIFEST)
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported storage format version {manifest['format_version']}."
        )
    classes = {
        "NCVGBDTRegressor": NCVGBDTRegressor,
        "NCVGBDTClassifier": NCVGBDTClassifier,
    }
    cls = classes[manifest["estimator"]]
    if _expected is not None and not issubclass(cls, _expected):
        raise TypeError(f"{path} holds a {cls.__name__}, not a {_expected.__name__}.")
    params = dict(manifest["params"], output_dir=str(path), resume=False)
    est = cls(**params)
    est.task_ = manifest["task"]
    est.feature_names_in_ = np.array(manifest["feature_names_in"], dtype=object)
    est.n_features_in_ = len(manifest["feature_names_in"])
    est.target_name_ = manifest["target_name"]
    est.objective_ = manifest["objective"]
    est.used_index_ = manifest["used_index"]
    est.n_samples_used_ = manifest["n_samples_used"]
    if manifest["classes"] is not None:
        est.classes_ = manifest["classes"]
    est._fingerprint = manifest["fingerprint"]
    keys = ("best_params", "scores", "explainations", "scores_sh", "explainations_sh")
    results = {k: [] for k in keys}
    for f in sorted((path / FOLDS_DIR).glob("fold_*.json")):
        fold = _json.load(f)
        for k in keys:
            results[k].append(fold[k])
    est.results_ = results
    if not manifest["complete"]:
        warnings.warn(
            f"{path} holds an incomplete run "
            f"({len(results['scores'])} folds). Continue it with "
            "resume=True."
        )
    current = _versions()
    differing = {
        k: (v, current.get(k))
        for k, v in manifest["versions"].items()
        if v != current.get(k)
    }
    if differing:
        warnings.warn(
            "Library versions differ from those used for these "
            f"results (stored, current): {differing}"
        )
    final = path / FINAL_MODEL
    if final.exists():
        est.best_estimator_ = joblib.load(final)
        est.best_params_ = est.best_estimator_.get_params()
    return est
