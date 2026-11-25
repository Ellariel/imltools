# Interpretable Machine Learning Toolbox

## ncvgbdt

Gradient Boosted Decision Trees (LightGBM) with **n**ested **c**ross-**v**alidation
and SHapley Additive exPlanations (SHAP), for tabular data with regression,
binary and multiclass targets, optionally with grouped observations (e.g. several
rows per participant).

This code is a reused version with modifications by Danila Valko
(d.v.valko[at]proton.me) of Dr. David Steyrl's codebase (david.steyrl[at]gmail.com).
Source code and description: https://github.com/univiemops/iml, one example here:
https://doi.org/10.31219/osf.io/um69t. The SHAP project: https://github.com/shap/shap.

You can cite this tool by mentioning one of Dr. Steyrl's works, e.g.
https://doi.org/10.1038/s41598-024-65088-z or https://doi.org/10.1371/journal.pone.0304285

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
from ncvgbdt import NCVGBDTRegressor, make_figures, load_model

model = NCVGBDTRegressor(
    x_con_names=["wt"],  # continuous predictors
    x_cat_bin_names=["vs"],  # binary predictors, coded 0/1
    x_cat_mult_names=["gear"],  # >2 categories, target encoded
    evaluation="CV",  # 'CV' | 'TT'
    output_dir="results/mpg",  # everything is stored here as JSON
)
model.fit(df, df["mpg"], groups=None)  # groups: e.g. df['participant']

model.cv_results_  # scores per outer fold, real vs. shuffled targets, tuned parameters
model.shap_effects_  # mean |SHAP| per fold and predictor (long format)
make_figures(model)  # figures into results/mpg/figures

model = load_model("results/mpg")  # later, in another session
```

Use `NCVGBDTClassifier` for categorical targets (two classes: binary, more:
multiclass; any labels). `fit(X, y, groups=None, test_idx=None)` follows the
scikit-learn convention; the estimators support `get_params`, `set_params` and
`clone`. See [example.py](example.py) (mtcars). Every parameter, attribute and
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
- **Complete cases.** Rows with a missing value in the target, the group variable
  or any predictor are dropped, separately for each target. Nothing is imputed
  (LightGBM could handle missing values; this tool does not use that).
- **Groups matter.** With repeated measures per person, pass `groups`.
  Without it every row is its own group and rows of the same person can end up in
  train and test set, which inflates performance. A `TT` split with
  `test_idx` that shares groups between train and test raises a warning.
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
  fold) and weight the confusion matrix plot. *Earlier versions used the class
  frequency itself as weight, which favours the majority class and contradicted
  the documentation; accuracies from those versions are not comparable.*
- The confusion matrix is class-weighted, so its diagonal sums to the balanced
  accuracy.

### Shuffled-target baseline and p values
- The baseline model reuses the hyperparameters tuned on the real data (it is not
  re-tuned on shuffled targets) and is refit on training targets that are shuffled
  **across rows, ignoring groups**. It estimates what the model does when there
  is no relationship between predictors and target.
- Figure p values come from a **one-sided corrected resampled t-test** (Nadeau &
  Bengio 2003) of the real model against the baseline over outer folds, assuming a
  test/train size ratio of 0.25 (5-fold CV). Positive difference = real model
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
- **Binary tasks**: main-effect SHAP values are stored with a class axis for
  class 0 and class 1; class 0 is the exact mirror image of class 1 (kept for
  symmetry with multiclass). Interaction values are stored for the positive
  class only.
- `shap_interactions=True` (the default is False) is slow and memory-hungry
  (features squared) and is **not supported for multiclass targets**
  (an error is raised before fitting). Switch it off for large problems.

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
- Only LightGBM. Upstream's TabPFN variant, huber loss and `shapiq` extensions
  are not ported.
- Very small data sets (like mtcars) work as a smoke test but give noisy
  estimates.

## Changes compared with the original code

The function `run_analysis` was replaced by scikit-learn style estimators
(`NCVGBDTRegressor`, `NCVGBDTClassifier`) with `fit(X, y, groups)`, fitted
attributes, JSON storage, `resume` and an optional final model. The parameters
changed accordingly (`type` -> `evaluation`, `g_name` -> `groups`,
`test_set_ind` -> `fit(test_idx=...)`, `def_test_size` -> `test_size`,
`objective` is inferred from the estimator and the target, `y_names` and
`skip_rows` are gone: select columns and rows in pandas). Results stored by the
old versions (pickle files) cannot be loaded. The code was also restructured
(split into `estimator` / `fitting` / `plotting`, `src` layout, installable
package). Bugs fixed and behaviours added:

- Seeding of all random steps (`seed` parameter); earlier versions were only
  partly seeded.
- `type` default was `'cv'` but the code compared with `'CV'`, so the default
  call failed; `evaluation` is now case-insensitive and validated. Mutable
  default arguments removed.
- Grouping crashed with a column name or a list of names; groups are now passed
  as an array.
- Without predictor lists the column order came from a Python `set` (random per
  process); order is now stable.
- Random test set for `TT` was drawn *with replacement* (duplicates, fewer test
  rows), could never include the last row, ignored groups and mixed row labels
  with positions. Now: group-disjoint random groups, positional indices.
- NumPy 2 compatibility of the group k-fold (`np.unique` inverse shape).
- Classification: balanced class weights (see "Scores"); target encoder uses
  `binary`/`multiclass` target type instead of treating class codes as numbers;
  multiclass SHAP aggregation; binary SHAP shape; confusion matrix.
- Corrected t-test: the p value used `|t|` and called itself one-sided; it is now
  truly one-sided.
- `make_figures` paired task and result files by directory listing order; it now
  takes the fitted estimator (or its directory). `ilm_shap_values_graph` could not exclude the first predictor.
- pandas 3 compatibility of the interaction figures; running means ignore NaN
  folds.
- Not ported from upstream (checked, no bug): SHAP tree-explainer patches
  mentioned in the upstream README were not needed with the tested versions.

## Development

```
uv run pytest            # about 5 minutes
```

`tests/test_example.py` re-runs the mtcars example and compares with reference
values (one set per numpy major version, with a small tolerance) and checks exact
reproducibility; `tests/test_modes.py` covers classification, grouped
train-test splits, interactions, validation and helpers; `tests/test_estimator.py`
covers the estimator interface, JSON storage, loading, resuming and the final
model. To test the older stack:

```
uv run --isolated --python 3.12 --with numpy==1.26.4 --with shap==0.46.0 --with scipy==1.12.0 --with scikit-learn==1.5.2 --with lightgbm==4.5.0 --with pandas==2.2.3 pytest
```
