# Interpretable Machine Learning Toolbox

## ncvgbdt

Gradient Boosted Decision Trees (LightGBM) with **n**ested **c**ross-**v**alidation
and SHapley Additive exPlanations (SHAP), for tabular data with regression,
binary and multiclass targets, optionally with grouped observations (e.g. several
rows per participant).

This code is a reused version with modifications by Danila Valko
(danila.valko[at]proton.me) of David Steyrl's codebase: https://github.com/univiemops/iml and the SHAP project: https://github.com/shap/shap.

The citation metadata of this package is in
[CITATION.cff](CITATION.cff); GitHub offers it under "Cite this repository".

## Install

```
uv sync                      # inside a clone of this repository (editable install)
uv add /path/to/imltools     # into another uv project
pip install /path/to/imltools
```

Python >= 3.12. It works with the pinned-era stack (numpy 1.26 / shap 0.46 /
scikit-learn 1.5 / lightgbm 4.5) and with current releases (numpy 2, shap 0.54,
...) on Python 3.12 and 3.13. **Numbers differ between these stacks** (see
"Reproducibility").

## Quick start

```python
import pandas as pd
from ncvgbdt import NCVGBDTRegressor, make_figures, load_model

# mtcars (in tests/data): 32 rows, only a smoke test, estimates are noisy
df = pd.read_csv("tests/data/mtcars.csv", index_col="model")

model = NCVGBDTRegressor(
    x_con_names=["wt"],  # continuous predictors
    x_cat_bin_names=["vs"],  # binary predictors, coded 0/1
    x_cat_mult_names=["gear"],  # >2 categories, target encoded
    evaluation="CV",  # 'CV' | 'TT'
    n_rep_outer_cv=1,  # quick demo; the default is 10 repetitions
    n_samples_rs=10,  # quick demo; the default is 100 random search draws
    output_dir="results/mpg",  # everything is stored here as JSON
)
model.fit(df, df["mpg"], groups=None)  # groups: e.g. df['participant']

model.cv_results_  # scores per outer fold, real vs. shuffled targets, tuned parameters
model.shap_effects_  # mean |SHAP| per fold and predictor (long format)
model.scores_summary_  # per metric: mean, sd, interval, shuffled baseline, p value
model.effect_sizes_  # per predictor: effect, share, direction (slope per SD)
make_figures(model)  # figures into results/mpg/figures

model = load_model("results/mpg")  # later, in another session
```

Use `NCVGBDTClassifier` for categorical targets (two classes: binary, more:
multiclass; any labels). `fit(X, y, groups=None, test_idx=None)` follows the
scikit-learn convention, and `y` and `groups` may also be column names of `X`
(`fit(df, 'mpg', groups='pid')`; the target and group columns are then never
used as predictors by default); the estimators support `get_params`, `set_params` and
`clone`. Every parameter, attribute and
method with options and defaults is documented in the docstrings, e.g.
`help(ncvgbdt.NCVGBDTRegressor)` and `help(ncvgbdt.make_figures)`.

`fit` runs the **evaluation**: nested cross-validation giving an honest
performance estimate and SHAP effects. It does not leave you with one "the
model": every outer fold has its own tuned model. If you also want to predict,
use `refit=True`: one more model is tuned and fitted on all rows and made available
as `best_estimator_`, `predict`, `predict_proba` and `explain(X)`. Its quality is
the one estimated by the outer folds; it is not evaluated itself. `refit` is off
by default because it costs one more hyperparameter search.

One estimator holds one target; loop over targets and use one `output_dir` each.

| Module | Content |
|---|---|
| `ncvgbdt.estimator` | `NCVGBDTRegressor`, `NCVGBDTClassifier`, `load_model`, storage |
| `ncvgbdt.fitting` | building blocks: nested CV, models, scoring, SHAP, group k-fold splitters |
| `ncvgbdt.plotting` | `make_figures` and all figure functions |

### Stored model (`output_dir`)

```
results/mpg/
  manifest.json          settings, library versions, data fingerprint, classes, ...
  folds/fold_0000.json   one file per outer fold: tuned parameters, scores,
  folds/fold_0001.json   predictions and SHAP values of the real and shuffled model
  final_model.joblib     only with refit=True
  figures/               created by make_figures
```

- Everything except the final model is **plain JSON**: readable from any
  language, no pickle, exact float round trip (NaN/Infinity are written as such,
  which Python reads but strict JSON parsers may not). SHAP values are stored
  as nested lists, so interaction values of big problems make large files.
- A fold file is written as soon as the fold is finished. If a long run is
  interrupted, call `fit` again with `resume=True`: finished folds are loaded
  and only the rest is computed. This only works with unchanged data and
  settings (otherwise an error); the result equals an uninterrupted run
  (tested).
- `final_model.joblib` is **pickle-based**: load it only from sources you trust,
  and expect it to fail on other scikit-learn / LightGBM versions. Everything
  else loads regardless. `load_model` warns when the library versions differ from
  the stored ones.
- Without `output_dir` nothing is written; use `model.save(path)` later.
- Fitting again with the same `output_dir` and `resume=False` removes the earlier
  results in that directory (only the files listed above). Use one directory per
  estimator.

## Method in brief

1. **Outer loop** (`evaluation='CV'`): `n_rep_outer_cv` repetitions of 5-fold
   cross-validation, folds formed from whole *groups*. `evaluation='TT'`: one
   group-disjoint train-test split (random, or the rows in `fit(test_idx=...)`).
2. **Inner loop** (training data of each outer fold only): randomized search over
   `colsample_bytree` ~ U(0.5, 1), `extra_trees` in {True, False} and
   `path_smooth` ~ log-U(1, 1000), `n_samples_rs` draws, scored by 5-fold group
   CV repeated until about `n_samples_inner_cv` predictions (1 to 10 repeats).
3. **Fixed** LightGBM settings: 100 leaves, learning rate 0.01, 1000 rounds, GOSS,
   `min_child_samples` 2, no L1/L2 penalty.
4. **Scoring** on the outer test fold: R2, MAE, MSE (regression) or balanced
   accuracy (classification).
5. **Baseline**: the tuned model is refit on shuffled training targets and scored
   and explained on the same test fold.
6. **SHAP** values on at most `max_samples_shap` random rows of each test fold.

## Design choices and caveats researchers should know

### Data and preprocessing
- **Missing values.** Rows with a missing target or group are always dropped.
  Missing *predictor* values are kept by default (`missing='native'`): nothing is
  imputed, LightGBM learns for every split which way missing values go, and the
  standardization ignores them. A predictor without any observed value raises an
  error. The default assumes that missingness is either uninformative or that its
  pattern is itself worth using: the model can use "missing" as information
  (whether that is wanted depends on the missingness mechanism), and SHAP
  attributes it to the predictor like any other value. A missing category of a
  target-encoded (multi-category) predictor is not a category of its own; it is
  encoded with the overall target mean, i.e. like an unknown category. With
  `missing='drop'` every row with any missing predictor value is removed
  (complete-case analysis). `used_index_` tells which rows were used; rows with
  missing values are skipped in the direction summaries of `effect_sizes_`
  (`slope_per_sd`, `corr_x_shap`).
- **What `groups` is for.** `groups` says which rows belong together: all rows
  with the same group value always end up on the same side of every split (outer
  folds, inner tuning folds and train-test split). Pass it whenever rows are not
  independent, e.g. several measurements per person; without it every row is its
  own group, rows of the same person can land in both train and test set, and the
  performance is inflated. A `TT` split with `test_idx` that shares groups between
  train and test raises a warning.
  - **The group is the unit the model is tested on.** With participants nested in
    countries, `groups=participant` asks "how well does the model predict new
    people?", while `groups=country` leaves whole countries out and asks "how well
    does it predict people from countries it has never seen?" (stricter; needs at
    least 5 countries, and a few very large countries make the folds uneven).
    Grouping by the coarser level keeps the finer level (participants) together too.
  - If the country should be a **predictor** (`x_cat_mult_names`), group by
    participant, not by country: with `groups=country` every test country is
    unseen and its category carries no information.
  - `groups` takes a single column; there are no nested levels.
- **Predictor types.** Continuous predictors are standardized. Binary predictors
  must already be numeric (0/1). Predictors with more than two categories are
  target encoded (scikit-learn `TargetEncoder`, `smooth='auto'`, internal 5-fold
  cross-fitting) *inside* the pipeline, i.e. fitted on training data only, so no
  target information leaks from the test fold. Only the list of categories is
  taken from the whole data set (so unseen categories in a fold do not fail).
  Ordinal predictors are best passed as continuous.
- If you pass no predictor lists at all, every column that is neither target nor
  group is used as a continuous predictor.
- The pipeline standardizes regression targets internally; scores and SHAP values
  are reported in original units.

### Scores
- **R2 is computed per outer test fold** and then averaged, not pooled over all
  predictions. On small folds R2 can be very negative; look at the distribution
  (violin plot), not only the mean.
- **Classification uses balanced accuracy**: every sample is weighted by
  `n / (n_classes * n_class)` (the scikit-learn `'balanced'` definition). The same
  weights are given to LightGBM as `class_weight` (frequencies of the training
  fold) and weight the confusion matrix plot.
- The confusion matrix is class-weighted, so its diagonal sums to the balanced
  accuracy.
- **More metrics** (all out of sample, per outer test fold, also for the
  shuffled-target model; `model.cv_results_`, summarized in
  `model.scores_summary_`):
  - Regression: R2, MAE, MSE, RMSE.
  - Classification: balanced accuracy (`acc`), AUC (binary; multiclass: macro
    one-vs-rest), log loss, Brier score, Matthews correlation (`mcc`), Cohen's
    `kappa`, macro F1, and pseudo R2 values: McFadden (`1 - ll_model / ll_null`),
    Cox-Snell, Nagelkerke (Cox-Snell scaled to a maximum of 1) and, for binary
    targets, Tjur (mean predicted probability of the positive class among
    positives minus among negatives).
  - **Probability based metrics are class-balanced**: samples are weighted by
    their class weight and the null model gives every class the same probability,
    because the models are trained with balanced class weights and so their
    probabilities reflect equal class priors. They are therefore not the same
    numbers as an unweighted log loss or McFadden R2 on the raw class
    frequencies. MCC and kappa use the labels unweighted; AUC is rank based.
  - Pseudo R2 values are computed on test data, so they can be **negative** (worse
    than the null model); they are not comparable to the R2 of a linear model
    and have no variance-explained interpretation.
  - **No AIC/BIC**: they need a likelihood and a parameter count, and a boosted
    tree ensemble has no meaningful parameter count. The out-of-sample log loss
    (equivalently the log-likelihood) from cross-validation is the proper
    analogue for comparing models.
  - The regression R2 uses the mean of the *test* fold as its baseline
    (scikit-learn's `r2_score`), not the mean of the training data.

### Effect sizes of predictors
`model.effect_sizes_` has one row per predictor (and class, for multiclass
targets), averaged over the outer folds, derived from the SHAP values of the test
rows:

- `effect`: mean absolute SHAP value, with an approximate 95 % interval
  (`effect_ci_low`, `effect_ci_high`); `effect_share`: percent of the total over
  predictors; `effect_shuffled`: the same for the shuffled-target model, i.e. the
  effect that arises without any real relation (a null reference);
  `effect_rel_target_sd` (regression): `effect` relative to the target's standard
  deviation. Units: target units (regression) or log-odds (classification).
- `slope_per_sd` (with interval) and `corr_x_shap`: **direction and a linear
  summary** of the effect: the slope of the SHAP values on the predictor times
  its standard deviation (about the change of the prediction per +1 SD of the
  predictor), and the correlation between predictor values and SHAP values.
  Both are NaN for multi-category predictors, which have no order.

Caveats: a boosted-tree model has no coefficients, so these are *summaries of a
possibly non-linear, interacting function*, not regression coefficients. A low
correlation can mean a non-monotonic effect, not "no effect". The effect is a
magnitude (non-negative); its interval is approximate (see below) and may reach
below zero. For binary targets the positive class is reported (the other class is
its mirror image). Effects of correlated predictors are shared between them (see
"SHAP"). Compare `effect` with `effect_shuffled` rather than with zero.

### Shuffled-target baseline and p values
- The baseline model reuses the hyperparameters tuned on the real data (it is not
  re-tuned on shuffled targets) and is refit on training targets that are shuffled
  **across rows, ignoring groups**. It estimates what the model does when there
  is no relationship between predictors and target.
- Figure p values and the `p_value`, `ci_low` and `ci_high` of
  `model.scores_summary_` come from a **one-sided corrected resampled t-test**
  (Nadeau & Bengio 2003) of the real model against the baseline over outer folds,
  assuming a test/train size ratio of 0.25 (5-fold CV); for error metrics (MAE,
  MSE, RMSE, log loss, Brier) a smaller value counts as better. Intervals of the
  effect sizes use the same corrected variance. Positive difference = real model
  better; a real model worse than the baseline is not significant.
- Folds of repeated CV are **not independent**, so these p values are
  approximate. They also shrink as `n_rep_outer_cv` grows although no new data
  came in. Treat them as a rough screen; report effect sizes and fold
  distributions. For `evaluation='TT'` there is only one fold and no meaningful p value.
- `add_multiple_comparison_correction=True` is a plain Bonferroni correction
  (p x number of predictors, for interactions x predictors squared). p values are
  rounded to 3 decimals before it is applied.

### SHAP
- Tree SHAP with `feature_perturbation='tree_path_dependent'` and no background
  data. This explains the **model**, not causal effects. Credit is shared among
  correlated predictors in a way that depends on how the trees split, so
  correlated predictors can look individually unimportant.
- Units: original target units (regression) or log-odds (classification). The
  "SHAP effect" is the mean absolute SHAP value. Effects are computed on a random
  subsample of each test fold (`max_samples_shap`, default 100); raise it for
  stable estimates if you have more rows.
- The effect of the shuffled model gives the null distribution for each
  predictor's effect.
- **Multiclass with target-encoded predictors:** the encoder creates one column
  per class; their SHAP values (and interaction values) are summed back to a
  single value per predictor (SHAP values are additive).
- **Multiclass interactions:** with `shap_interactions=True` the interaction
  values are stored per class (rows x features x features x classes). The
  interaction figures and all other SHAP figures are drawn once per class (file
  names end in `_class_<label>`), as for a binary target. Summing the interaction
  values of a feature over its partners gives its main SHAP value, so the effects
  equal those of a run without interactions (tested).
- **Binary tasks**: main-effect SHAP values are stored with a class axis for
  class 0 and class 1; class 0 is the exact mirror image of class 1 (kept for
  symmetry with multiclass). Interaction values are stored for the positive
  class only.
- `shap_interactions=True` (the default is False) is slow and memory-hungry
  (features squared, and times the number of classes for multiclass targets).
  Leave it off for large problems.

### Reproducibility
- `seed` (default 3141592) fixes every random step: outer and inner splits,
  hyperparameter draws, target encoder, LightGBM, SHAP subsample, shuffled
  targets, random test groups. Two runs with the same seed, same data and same
  library versions give identical results (tested). `seed=None` makes tuning and
  models unseeded.
- **Results differ between library versions** (e.g. numpy 1 / scikit-learn 1.5 vs.
  current releases) even with the same seed, because scikit-learn and LightGBM
  change their random number handling. Record versions with your results
  (`uv.lock`, `pip freeze`).
- Fold assignment depends on the order of the group labels; changing the sort
  order of groups changes folds, hence scores.

### Limitations
- Fixed 5 outer and inner folds; no sample weights; no time-series splits.
- Hyperparameter search covers three parameters only (see above).
- Only LightGBM, with squared-error loss for regression (see the comparison with
  upstream below).
- Very small data sets (like mtcars) work as a smoke test but give noisy
  estimates.

## Differences from the original (univiemops/iml)

This package was derived from an earlier version of Dr. Steyrl's
[iml](https://github.com/univiemops/iml) scripts and has **not** been merged with
its later development. The comparison below is with the upstream state of
6 July 2026 (commit `f565183`: `iml_2_mdl_lgbm.py`, `iml_3_plt.py`,
`sklearn_repeated_group_k_fold.py`). Check upstream when you need the original
behaviour; results of the two are not expected to be numerically identical.

### Same as upstream
Group-wise repeated 5-fold outer CV with an inner randomized search on the
training data; fixed 100 leaves, learning rate 0.01 and 1000 rounds; classification
scored by balanced accuracy with balanced class weights; target encoding of
multi-category predictors inside the pipeline (binary/multiclass target type for
classification, SHAP values summed back per predictor); shuffled-target baseline
that reuses the tuned hyperparameters; tree SHAP without background data;
one-sided corrected resampled t-test (signed t statistic) with optional Bonferroni
correction; mean absolute SHAP value as predictor effect; missing predictor values
handled by LightGBM; SHAP interaction values for multiclass targets (stored per class
and aggregated over target-encoded columns).

### Form and interface
- **Package instead of scripts.** Upstream is a set of scripts configured by
  editing `main()` (Excel input, log files, pickle output). Here: an installable
  package (`ncvgbdt/` folder, `pyproject.toml`, uv) with scikit-learn style estimators
  (`NCVGBDTRegressor`, `NCVGBDTClassifier`), `fit(X, y, groups)`, fitted
  attributes, `get_params` / `clone`, `y` and `groups` also by column name.
- **Storage.** JSON per outer fold plus a manifest (settings, library versions,
  data fingerprint) instead of pickles; `load_model`, `resume` of interrupted runs,
  `save`. Upstream's pickles also hold the fitted pipeline of every outer fold, the
  test predictors and test row indices; **this package does not store fold models or
  per-fold row indices** (only predictions, scores, parameters and SHAP values).
- **Optional final model** (`refit=True`): `predict`, `predict_proba`,
  `explain(X)`. Upstream has no equivalent.
- **Tests and documentation:** an automated test suite (reference values, exact
  reproducibility, classification, grouped splits, storage, resuming), docstrings
  with all options and defaults, and this README.
- **Library versions.** Tested on Python 3.12 and 3.13 with current releases
  (numpy 2, scikit-learn, shap 0.54, pandas 3) and with the older numpy 1.26 /
  shap 0.46 / scikit-learn 1.5 stack. The shap source patches described in
  upstream's `Readme.txt` were not needed for the configurations supported here.

### Reproducibility
Upstream fixes the outer split and the SHAP subsample (`random_state=1000`), but
the inner CV, the parameter search, the target encoder, LightGBM, the shuffled
targets and (for a train-test split) the row subsampling are unseeded. Here one
`seed` determines every random step; the same seed, data and library versions
give identical results (tested).

### Model, tuning and sampling settings
| | upstream | this package |
|---|---|---|
| Regression loss | Huber | squared error |
| Row sampling | bagging | GOSS (`top_rate` 0.5) |
| `min_child_samples` / `min_split_gain` / `min_child_weight` | 1 / 0 / 1e-5 | 2 / 1e-6 / 1e-6 |
| `colsample_bytree` search | U(0.1, 1.0) | U(0.5, 1.0) |
| `path_smooth` search | log-U(1, 100) | log-U(1, 1000) |
| Inner search score (regression) | R2 | negative MSE |
| Binary target | multiclass objective with 2 classes | binary objective |
| Outer repetitions | derived from `N_PRED_OUTER_CV` (default 10000), at least 3 | `n_rep_outer_cv` (default 10) |
| Inner repetitions | from `N_PRED_INNER_CV` (default 1000), 3 to 10 | from `n_samples_inner_cv` (default 10000), 1 to 10 |
| SHAP rows | `N_SAMPLES_SHAP` (default 1000) in total, spread over the folds | `max_samples_shap` per outer fold (default 100) |
| Train-test split rows | train and test subsampled to at most 10000 rows each | no subsampling |
| SHAP interaction values | always computed | optional (default off) |

### Evaluation and reporting
- **Train-test split (`evaluation='TT'`):** a random test set of whole groups
  (`test_size`) or the rows given in `test_idx`. Upstream takes row indices from
  `randint.rvs(...)` in the configuration, which draws with replacement (duplicate
  indices) and ignores groups.
- **More metrics:** RMSE; for classification AUC, log loss, Brier score, MCC, kappa,
  macro F1 and the pseudo R2 values McFadden, Cox-Snell, Nagelkerke and Tjur.
  Upstream reports R2 and MAE (regression) and balanced accuracy (classification).
- **Summaries:** `scores_summary_` (mean, sd, approximate interval, baseline,
  p value per metric), `shap_effects_` and `effect_sizes_` (magnitude, share,
  relative size, null reference, standardized slope and correlation as direction).
- **Binary SHAP layout:** main-effect values carry a class axis (class 0 is the
  mirror of class 1) so that the figures treat binary and multiclass alike.
- **Figures:** `make_figures` takes a fitted estimator or its directory. The
  plotting code stems from the earlier upstream version and was **not synced with
  upstream's later rewrite of `iml_3_plt.py`**, so the set and look of the figures
  differ; adjusted for current library versions (pandas 3) and the new layout.

### Not supported or not ported
- **Missing values:** as upstream, missing predictor values are left to LightGBM by
  default; `missing='drop'` removes such rows instead. Rows with a missing target
  are dropped in both; rows with a missing group are dropped here.
- **Shap source patches:** upstream's `Readme.txt` describes two patches of shap's
  `_tree.py`. They were not needed for the supported configurations, including
  multiclass interaction values, with shap 0.46 and 0.54.
- Upstream's other scripts: the exploratory data analysis (`iml_1_eda.py`) and the
  TabPFN variant (`iml_2_mdl_tabpfn.py`); and the `shapiq` dependency.
- Fixed 5 outer and inner folds (as upstream), and no tuning beyond the three
  parameters listed above.

## Development

```
uv run pytest            # about 5 minutes
```

`tests/test_example.py` re-runs the mtcars example (`tests/data/mtcars.csv`) and compares with reference
values (one set per numpy major version, with a small tolerance) and checks exact
reproducibility; `tests/test_modes.py` covers classification, grouped
train-test splits, interactions, validation and helpers; `tests/test_estimator.py`
covers the estimator interface, JSON storage, loading, resuming and the final
model. To test the older stack:

```
uv run --isolated --python 3.12 --with numpy==1.26.4 --with shap==0.46.0 --with scipy==1.12.0 --with scikit-learn==1.5.2 --with lightgbm==4.5.0 --with pandas==2.2.3 pytest
```
