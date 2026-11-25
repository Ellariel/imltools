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
                     make_figures)
from ncvgbdt.fitting import (  # noqa: E402
    aggregate_shap_explanations, get_class_w, get_te_target_type,
    get_test_groups)

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
    with pytest.raises(ValueError, match="multiclass"):
        NCVGBDTClassifier(shap_interactions=True, **FAST).fit(
            X, data['multi'])
    with pytest.raises(ValueError, match="two classes"):
        NCVGBDTClassifier(shap_interactions=False, **FAST).fit(
            X, np.zeros(len(X)))


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


def test_missing_values_are_dropped(data):
    X = data[['a', 'b', 'c']].copy()
    X.loc[3, 'a'] = np.nan
    y = data['y'].copy()
    y.loc[7] = np.nan
    model = NCVGBDTRegressor(shap_interactions=False, **FAST)
    model.fit(X, y, groups=data['pid'])
    assert model.n_samples_used_ == len(X) - 2
    assert 3 not in model.used_index_ and 7 not in model.used_index_


def test_target_and_groups_by_column_name(data):
    """Names and arrays give identical results; the target and the group
    column are never used as predictors by default."""
    sub = data[['pid', 'a', 'b', 'y']]
    by_name = NCVGBDTRegressor(shap_interactions=False, **FAST).fit(
        sub, 'y', groups='pid')
    by_array = NCVGBDTRegressor(shap_interactions=False, **FAST).fit(
        sub, sub['y'], groups=sub['pid'])
    for m in (by_name, by_array):
        assert list(m.feature_names_in_) == ['a', 'b']
        assert m.target_name_ == 'y'
    pd.testing.assert_frame_equal(by_name.cv_results_, by_array.cv_results_)


def test_column_name_errors(data):
    sub = data[['pid', 'a', 'b', 'y']]
    with pytest.raises(KeyError, match="'nope' given as y"):
        NCVGBDTRegressor(**FAST).fit(sub, 'nope')
    with pytest.raises(KeyError, match="'nope' given as groups"):
        NCVGBDTRegressor(**FAST).fit(sub, 'y', groups='nope')
    with pytest.raises(ValueError, match="leak"):
        NCVGBDTRegressor(x_con_names=['a', 'y'], **FAST).fit(sub, 'y')
    with pytest.raises(ValueError, match="leak"):
        NCVGBDTRegressor(x_con_names=['a', 'y'], **FAST).fit(sub, sub['y'])
