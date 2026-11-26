"""Tests for classification, grouped train-test split, validation and helpers.

Fast unit tests first, slower end-to-end runs (fit + figures) afterwards. All
data is synthetic, no network access is needed.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shap import Explanation

import matplotlib
matplotlib.use("Agg")

from ncvgbdt import (NCVGBDTClassifier, NCVGBDTRegressor,  # noqa: E402
                     load_model, make_figures)
from ncvgbdt.fitting import (  # noqa: E402
    aggregate_shap_explanations, classification_metrics, get_class_w,
    get_te_target_type, get_test_groups, metrics_for)

FAST = dict(n_rep_outer_cv=1, n_samples_inner_cv=10, n_samples_rs=5,
            max_samples_shap=10)


@pytest.fixture(scope="module")
def data():
    """150 rows, 50 groups of 3 rows, regression and class targets."""
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({'pid': np.repeat(np.arange(50), 3),
                       'a': rng.normal(size=n),
                       'b': rng.normal(size=n),
                       'c': rng.integers(0, 4, n)})
    df['y'] = df.a * 2 + df.c + rng.normal(size=n)
    df['bin'] = (df.a + rng.normal(size=n) > 0).astype(int)
    df['multi'] = pd.cut(df.a + 0.5 * df.c + rng.normal(size=n),
                         [-99, -0.5, 1.5, 99], labels=False)
    return df


# --- unit tests -------------------------------------------------------------

def test_class_weights_are_inverse_frequency():
    w = get_class_w(np.array([0] * 80 + [1] * 20))
    assert w[1] > w[0]
    # Same definition as scikit-learn's class_weight='balanced'
    assert w[0] == pytest.approx(100 / (2 * 80))
    assert w[1] == pytest.approx(100 / (2 * 20))
    # Balanced data -> all weights 1
    assert set(get_class_w(np.array([0, 1, 2] * 5)).values()) == {1.0}


def test_figure_file_names_are_not_cut_by_the_directory():
    """File names used to be truncated together with the directory, so long
    output paths made different figures overwrite each other."""
    from ncvgbdt.plotting import _save_path
    long_dir = '/very/long/' + 'x' * 200
    a = _save_path(long_dir, 'analysis_CV_5_y_class_0_shap_effects')
    b = _save_path(long_dir, 'analysis_CV_5_y_class_1_shap_effects')
    assert a.startswith(long_dir + '/') and a != b
    # a very long file name is shortened, but stays unique
    c = _save_path('d', 'z' * 300 + 'a')
    d = _save_path('d', 'z' * 300 + 'b')
    assert len(c) < 140 and c != d


def test_te_target_type():
    assert get_te_target_type('regression') == 'continuous'
    assert get_te_target_type('binary') == 'binary'
    assert get_te_target_type('multiclass') == 'multiclass'
    with pytest.raises(ValueError):
        get_te_target_type('nonsense')


@pytest.mark.parametrize('interactions', [False, True])
def test_aggregate_shap_explanations(interactions):
    rng = np.random.default_rng(1)
    names = ['x', 'g_0', 'g_1', 'g_2', 'z']
    shape = (7, 5, 5, 3) if interactions else (7, 5, 3)
    values = rng.normal(size=shape)
    exp = Explanation(values, base_values=np.zeros((7, 3)),
                      data=np.zeros((7, 5)), feature_names=names)
    out = aggregate_shap_explanations(
        exp, {'g': ['g_0', 'g_1', 'g_2']}, interactions=interactions)
    # Group sits where its first member was
    assert out.feature_names == ['x', 'g', 'z']
    # Total contribution per sample and class is preserved (additivity)
    axes = (1, 2) if interactions else (1,)
    np.testing.assert_allclose(out.values.sum(axis=axes),
                               values.sum(axis=axes))
    # Group value is the sum of its members
    if not interactions:
        np.testing.assert_allclose(out.values[:, 1], values[:, 1:4].sum(1))
    with pytest.raises(ValueError):
        aggregate_shap_explanations(exp, {'g': ['missing']})


def test_get_test_groups_keeps_groups_together():
    g = pd.DataFrame({'pid': np.repeat(np.arange(50), 3)})
    i_tst = get_test_groups(g, 0.2, seed=1)
    labels = g['pid'].to_numpy()
    in_test = set(labels[i_tst])
    in_train = set(np.delete(labels, i_tst))
    assert not in_test & in_train
    assert len(i_tst) >= 30
    # reproducible
    np.testing.assert_array_equal(i_tst, get_test_groups(g, 0.2, seed=1))
    assert not np.array_equal(i_tst, get_test_groups(g, 0.2, seed=2))


def test_get_test_groups_without_groups_is_row_subset():
    g = pd.DataFrame({'idx': np.arange(100)})
    i_tst = get_test_groups(g, 0.25, seed=0)
    assert len(i_tst) == 25
    assert len(set(i_tst)) == 25  # no duplicates


def test_validation_errors(data):
    X = data[['a', 'b', 'c']]
    with pytest.raises(ValueError, match="evaluation must be"):
        NCVGBDTRegressor(evaluation='nope', **FAST).fit(X, data['y'])
    with pytest.raises(KeyError, match="missing_col"):
        NCVGBDTRegressor(x_con_names=['missing_col'], **FAST).fit(X, data['y'])
    with pytest.raises(TypeError, match="DataFrame"):
        NCVGBDTRegressor(**FAST).fit(X.to_numpy(), data['y'])
    with pytest.raises(TypeError, match="numeric"):
        NCVGBDTRegressor(**FAST).fit(X, data['y'].astype(str))
    with pytest.raises(ValueError, match="more than once"):
        NCVGBDTRegressor(x_con_names=['a'], x_cat_bin_names=['a'],
                         **FAST).fit(X, data['y'])
    with pytest.raises(ValueError, match="different lengths"):
        NCVGBDTRegressor(**FAST).fit(X, data['y'].iloc[:-1])
    with pytest.raises(ValueError, match="resume"):
        NCVGBDTRegressor(resume=True, **FAST).fit(X, data['y'])
    with pytest.raises(ValueError, match="missing must be"):
        NCVGBDTRegressor(missing='nope', **FAST).fit(X, data['y'])
    all_nan = X.assign(a=np.nan)
    with pytest.raises(ValueError, match="without any observed value"):
        NCVGBDTRegressor(**FAST).fit(all_nan, data['y'])
    with pytest.raises(ValueError, match="two classes"):
        NCVGBDTClassifier(shap_interactions=False, **FAST).fit(
            X, np.zeros(len(X)))


def _labels_and_proba(kind):
    y = np.array([0] * 30 + [1] * 10)
    if kind == 'perfect':
        p1 = np.where(y == 1, 0.999, 0.001)
    else:
        p1 = np.full(len(y), 0.5)
    return y, np.column_stack([1 - p1, p1])


def test_classification_metrics_reference_points():
    for kind, expected in (('perfect', 1.0), ('uniform', 0.0)):
        y, proba = _labels_and_proba(kind)
        m = classification_metrics(y, proba.argmax(1), proba, [0, 1],
                                   get_class_w(y))
        assert set(m) == set(metrics_for('binary'))
        assert m['mcfadden'] == pytest.approx(expected, abs=2e-3)
        assert m['nagelkerke'] == pytest.approx(expected, abs=2e-3)
        assert m['tjur'] == pytest.approx(expected, abs=5e-3)
        assert m['auc'] == pytest.approx(0.5 + expected / 2)
    # a perfect classifier: errors vanish, a uniform one gets the null values
    y, proba = _labels_and_proba('perfect')
    m = classification_metrics(y, proba.argmax(1), proba, [0, 1],
                               get_class_w(y))
    assert m['acc'] == m['mcc'] == m['kappa'] == m['f1_macro'] == 1.0
    assert m['brier'] < 1e-5 and m['logloss'] < 2e-3
    y, proba = _labels_and_proba('uniform')
    m = classification_metrics(y, proba.argmax(1), proba, [0, 1],
                               get_class_w(y))
    assert m['logloss'] == pytest.approx(np.log(2))
    assert m['brier'] == pytest.approx(0.25)


def test_classification_metrics_are_class_balanced():
    """A model that always predicts the majority class gets balanced accuracy
    0.5 and no better than null probability metrics."""
    y = np.array([0] * 90 + [1] * 10)
    proba = np.column_stack([np.full(100, 0.9), np.full(100, 0.1)])
    m = classification_metrics(y, proba.argmax(1), proba, [0, 1],
                               get_class_w(y))
    assert m['acc'] == pytest.approx(0.5)
    assert m['mcfadden'] <= 0           # worse than equal probabilities


def test_classification_metrics_multiclass_and_string_labels():
    rng = np.random.default_rng(0)
    classes = np.array(['a', 'b', 'c'])
    y = classes[rng.integers(0, 3, 90)]
    proba = np.full((90, 3), 1 / 3)
    m = classification_metrics(y, np.full(90, 'a'), proba, classes,
                               get_class_w(y))
    assert np.isnan(m['tjur'])
    assert m['logloss'] == pytest.approx(np.log(3))
    assert m['mcfadden'] == pytest.approx(0.0, abs=1e-12)
    assert m['brier'] == pytest.approx(2 / 3)
    assert m['auc'] == pytest.approx(0.5)
    # unseen test label: probability metrics undefined, label metrics fine
    m = classification_metrics(np.array(['z', 'a']), np.array(['a', 'a']),
                               proba[:2], classes, {'z': 1.0, 'a': 1.0})
    assert np.isnan(m['logloss']) and m['acc'] == 0.5


# --- end-to-end runs --------------------------------------------------------

@pytest.fixture(scope="module")
def multiclass_run(data, tmp_path_factory):
    """Multiclass CV with a target-encoded multi-category predictor."""
    d = tmp_path_factory.mktemp("multiclass")
    model = NCVGBDTClassifier(x_con_names=['a', 'b'], x_cat_mult_names=['c'],
                              shap_interactions=False, output_dir=d, **FAST)
    model.fit(data, data['multi'], groups=data['pid'])
    return d, model


def test_multiclass_target_encoding_and_shap(multiclass_run):
    _, model = multiclass_run
    assert list(model.classes_) == [0, 1, 2]
    exp = model.explanations_[0]
    # One SHAP column per predictor (not one per class of the encoder)
    assert exp.feature_names == ['a', 'b', 'c']
    assert exp.values.shape[1:] == (3, 3)  # features x classes
    assert exp.data.shape[1] == 3
    # Balanced accuracy of a working model beats the shuffled baseline
    assert 0 <= model.score_shuffled_ <= 1
    assert model.score_ > model.score_shuffled_
    # Effects in long format, one entry per class
    eff = model.shap_effects_
    assert set(eff['class']) == {0, 1, 2}
    assert set(eff['feature']) == {'a', 'b', 'c'}
    # Metrics: everything but the binary-only Tjur R2
    cv = model.cv_results_
    for m in metrics_for('multiclass'):
        assert m in cv.columns and m + '_shuffled' in cv.columns
    assert 'tjur' not in cv.columns
    assert cv['auc'].between(0, 1).all()
    assert (cv['nagelkerke'] <= 1).all()
    # Effect sizes per predictor and class; direction is undefined for the
    # target-encoded predictor 'c'
    es = model.effect_sizes_
    assert len(es) == 3 * 3
    assert es[es['feature'] == 'c']['slope_per_sd'].isna().all()
    assert es[es['feature'] != 'c']['slope_per_sd'].notna().all()
    for cls, grp in es.groupby('class'):
        assert grp['effect_share'].sum() == pytest.approx(100.0)


def test_multiclass_figures(multiclass_run):
    d, model = multiclass_run
    out = make_figures(model, figures_dir=d / 'figs')
    assert len(list(Path(out).glob('*.png'))) >= 10


def test_binary_grouped_train_test_split_with_interactions(data, tmp_path):
    """Lower-case evaluation, group-disjoint random test set, interactions."""
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*both train and test.*')
        model = NCVGBDTClassifier(
            x_con_names=['a', 'b'], x_cat_mult_names=['c'], evaluation='tt',
            shap_interactions=True, output_dir=tmp_path, **FAST)
        model.fit(data, data['bin'], groups=data['pid'])
    test_rows = model.task_['TEST_SET_IND_USED']
    groups = data['pid'].to_numpy()
    assert not set(groups[test_rows]) & set(np.delete(groups, test_rows))
    assert len(model.cv_results_) == 1
    assert list(model.classes_) == [0, 1]
    # binary interactions: (n, features, features)
    assert model.explanations_[0].values.shape[1:] == (3, 3)
    out = make_figures(model, figures_dir=tmp_path / 'figs')
    assert len(list(Path(out).glob('*.png'))) >= 5


def test_default_arguments_run(data):
    """Defaults: CV, all columns continuous, no groups; stable column order."""
    model = NCVGBDTRegressor(shap_interactions=False, **FAST)
    model.fit(data[['a', 'b']], data['y'])
    assert list(model.feature_names_in_) == ['a', 'b']
    assert model.explanations_[0].feature_names == ['a', 'b']
    assert model.target_name_ == 'y'


def _with_missing(data):
    """Predictors of every kind with missing values and one missing target."""
    rng = np.random.default_rng(3)
    df = data[['pid', 'a', 'b', 'c', 'y']].copy()
    df['d'] = rng.integers(0, 2, len(df)).astype(float)
    for col, frac in (('a', .15), ('b', .1), ('c', .15), ('d', .1)):
        df.loc[rng.random(len(df)) < frac, col] = np.nan
    df.loc[7, 'y'] = np.nan
    return df


def test_missing_predictors_drop_mode(data):
    df = _with_missing(data)
    names = dict(x_con_names=['a', 'b'], x_cat_bin_names=['d'],
                 x_cat_mult_names=['c'])
    model = NCVGBDTRegressor(missing='drop', shap_interactions=False,
                             **names, **FAST).fit(df, 'y', groups='pid')
    complete = df[['a', 'b', 'c', 'd', 'y']].notna().all(axis=1)
    assert model.n_samples_used_ == int(complete.sum())
    np.testing.assert_array_equal(model.used_index_,
                                  np.flatnonzero(complete))


def test_missing_predictors_native_mode(data, tmp_path):
    """Default: only rows with a missing target are dropped; the model, SHAP,
    the final model, storage and figures cope with missing predictors."""
    df = _with_missing(data)
    names = dict(x_con_names=['a', 'b'], x_cat_bin_names=['d'],
                 x_cat_mult_names=['c'])
    model = NCVGBDTRegressor(refit=True, shap_interactions=True,
                             output_dir=tmp_path, **names, **FAST)
    model.fit(df, 'y', groups='pid')
    assert model.missing == 'native'
    assert model.n_samples_used_ == len(df) - 1
    assert 7 not in model.used_index_
    assert df[['a', 'b', 'c', 'd']].isna().any(axis=1).sum() > 30
    # the explained data still contain the missing values
    shown = np.asarray(model.explanations_[0].data, dtype=float)
    assert np.isnan(shown).any()
    # effect sizes use the observed rows; direction undefined for nominal 'c'
    es = model.effect_sizes_.set_index('feature')
    assert es['effect'].notna().all()
    assert es.loc['a', 'corr_x_shap'] > 0.5
    assert np.isnan(es.loc['c', 'slope_per_sd'])
    # the final model predicts and explains rows with missing values
    rows = df[df[['a', 'b', 'c', 'd']].isna().any(axis=1)].head(5)
    assert np.isfinite(model.predict(rows)).all()
    assert model.explain(rows).values.shape[0] == 5
    # round trip keeps everything, including the missing values
    loaded = load_model(tmp_path)
    pd.testing.assert_frame_equal(loaded.effect_sizes_, model.effect_sizes_)
    np.testing.assert_array_equal(
        np.isnan(np.asarray(loaded.explanations_[0].data, dtype=float)),
        np.isnan(shown))
    out = make_figures(loaded, figures_dir=tmp_path / 'figs')
    assert len(list(Path(out).glob('*.png'))) >= 10


def test_multiclass_shap_interactions(data, tmp_path):
    """Interaction values per class; their sum over the second feature axis
    equals the main SHAP values, so effects match a run without them."""
    kw = dict(x_con_names=['a', 'b'], x_cat_mult_names=['c'], **FAST)
    kw.pop('max_samples_shap')
    inter = NCVGBDTClassifier(shap_interactions=True, max_samples_shap=12,
                              output_dir=tmp_path, **kw)
    inter.fit(data, 'multi', groups='pid')
    main = NCVGBDTClassifier(shap_interactions=False, max_samples_shap=12,
                             **kw).fit(data, 'multi', groups='pid')
    # (rows, features, features, classes), target encoding aggregated
    assert inter.explanations_[0].values.shape == (12, 3, 3, 3)
    assert inter.explanations_[0].feature_names == ['a', 'b', 'c']
    keys = ['fold', 'class', 'feature']
    a = inter.shap_effects_.set_index(keys)
    b = main.shap_effects_.set_index(keys)
    np.testing.assert_allclose(a['effect'], b['effect'], atol=1e-8)
    assert len(inter.effect_sizes_) == 3 * 3
    # one set of SHAP figures per class
    out = make_figures(inter, figures_dir=tmp_path / 'figs')
    names = [p.name for p in Path(out).glob('*.png')]
    for label in inter.classes_:
        assert any(f'_class_{label}_' in n and 'inter' in n for n in names)
    assert len(names) >= 3 * 10
