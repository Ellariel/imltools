"""Tests of the estimator interface: parameters, storage (JSON), loading,
resuming and the final model (refit)."""
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shap import Explanation
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

import matplotlib
matplotlib.use("Agg")

from ncvgbdt import (NCVGBDTClassifier, NCVGBDTRegressor,  # noqa: E402
                     load_model, make_figures)
from ncvgbdt import _json  # noqa: E402

FAST = dict(n_rep_outer_cv=1, n_samples_inner_cv=10, n_samples_rs=5,
            max_samples_shap=10, shap_interactions=False)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({'pid': np.repeat(np.arange(50), 3),
                       'a': rng.normal(size=n),
                       'b': rng.normal(size=n),
                       'c': rng.integers(0, 4, n)})
    df['y'] = df.a * 2 + df.c + rng.normal(size=n)
    df['bin'] = np.where(df.a + rng.normal(size=n) > 0, 'yes', 'no')
    return df


def _regressor(out=None, **kw):
    params = dict(x_con_names=['a', 'b'], x_cat_mult_names=['c'],
                  output_dir=out, **FAST)
    params.update(kw)
    return NCVGBDTRegressor(**params)


@pytest.fixture(scope="module")
def fitted(data, tmp_path_factory):
    d = tmp_path_factory.mktemp("fitted")
    model = _regressor(d, refit=True).fit(data, data['y'],
                                          groups=data['pid'])
    return d, model


# --- JSON codec ---------------------------------------------------------------

def test_json_roundtrip(tmp_path):
    exp = Explanation(np.random.default_rng(0).normal(size=(4, 3, 2)),
                      base_values=np.zeros((4, 2)), data=np.ones((4, 3)),
                      feature_names=['a', 'b', 'c'], compute_time=1.5)
    obj = {'arr': np.arange(6).reshape(2, 3),
           'f': np.array([0.1, np.nan, 1 / 3]),
           'strs': np.array(['x', 'yy']),
           'weights': {0: 1.5, 1: 0.25},            # non-str keys survive
           'named': {'no': 1.0, 'yes': 2.0},
           'np_scalar': np.float64(0.1) + np.float64(0.2),
           'none': None, 'exp': exp, 'nested': [(1, 2), {'k': np.int64(3)}]}
    _json.dump(obj, tmp_path / 'x.json')
    back = _json.load(tmp_path / 'x.json')
    np.testing.assert_array_equal(back['arr'], obj['arr'])
    assert back['arr'].dtype == obj['arr'].dtype
    np.testing.assert_array_equal(back['f'], obj['f'])      # NaN equal
    assert list(back['strs']) == ['x', 'yy']
    assert back['weights'] == {0: 1.5, 1: 0.25}
    assert back['named'] == {'no': 1.0, 'yes': 2.0}
    assert back['np_scalar'] == obj['np_scalar']             # exact float
    assert back['none'] is None
    np.testing.assert_array_equal(back['exp'].values, exp.values)
    assert back['exp'].feature_names == ['a', 'b', 'c']
    assert back['exp'].compute_time == 1.5
    assert back['nested'] == [[1, 2], {'k': 3}]


def test_json_rejects_unknown_objects(tmp_path):
    with pytest.raises(TypeError):
        _json.dump({'x': object()}, tmp_path / 'x.json')


# --- sklearn interface ----------------------------------------------------------

def test_params_clone_and_set_params():
    m = _regressor(seed=7, refit=True)
    params = m.get_params()
    assert params['seed'] == 7 and params['refit'] is True
    assert params['x_con_names'] == ['a', 'b']
    c = clone(m)
    assert c.get_params() == params and c is not m
    m.set_params(n_rep_outer_cv=3)
    assert m.n_rep_outer_cv == 3 and c.n_rep_outer_cv == 1
    assert 'NCVGBDTRegressor' in repr(m)


def test_not_fitted_errors(data):
    m = _regressor()
    for attr in ('cv_results_', 'shap_effects_', 'score_', 'explanations_'):
        with pytest.raises(NotFittedError):
            getattr(m, attr)
    with pytest.raises(NotFittedError):
        make_figures(m)
    with pytest.raises(NotFittedError):
        m.save('somewhere')
    # attribute access via hasattr must not blow up (NotFittedError is an
    # AttributeError)
    assert not hasattr(m, 'cv_results_')


def test_fit_returns_self_and_keeps_params(data):
    m = _regressor()
    before = m.get_params()
    assert m.fit(data, data['y']) is m
    assert m.get_params() == before
    assert m.n_features_in_ == 3
    assert list(m.feature_names_in_) == ['a', 'b', 'c']


def test_no_final_model_without_refit(data):
    m = _regressor().fit(data, data['y'])
    assert not hasattr(m, 'best_estimator_')
    with pytest.raises(NotFittedError, match='refit=True'):
        m.predict(data)
    with pytest.raises(NotFittedError, match='refit=True'):
        m.explain(data)


# --- results and storage ----------------------------------------------------------

def test_fitted_attributes(fitted):
    _, m = fitted
    cv = m.cv_results_
    assert len(cv) == 5
    for col in ('fold', 'score', 'score_shuffled', 'r2', 'mae', 'mse',
                'param_colsample_bytree', 'param_extra_trees',
                'param_path_smooth'):
        assert col in cv.columns
    assert m.score_ == pytest.approx(np.nanmean(cv['score']))
    assert m.score_ > m.score_shuffled_
    eff = m.shap_effects_
    assert set(eff.columns) == {'fold', 'class', 'feature', 'effect',
                                'effect_shuffled'}
    assert len(eff) == 5 * 3
    # the informative predictor 'a' beats the pure noise predictor 'b'
    mean = eff.groupby('feature')['effect'].mean()
    assert mean['a'] > mean['b']


def test_scores_summary(fitted):
    _, m = fitted
    summ = m.scores_summary_
    assert list(summ.index) == ['r2', 'mae', 'mse', 'rmse']
    assert summ.loc['r2', 'mean'] == pytest.approx(m.score_)
    assert (summ.loc['r2', 'ci_low'] < summ.loc['r2', 'mean'] <
            summ.loc['r2', 'ci_high'])
    # errors: lower is better, so the real model beats the baseline
    assert summ.loc['rmse', 'mean'] < summ.loc['rmse', 'mean_shuffled']
    assert (summ['p_value'] < 0.05).all()
    assert m.cv_results_['rmse'].tolist() == pytest.approx(
        np.sqrt(m.cv_results_['mse']).tolist())


def test_effect_sizes_regression(fitted):
    _, m = fitted
    es = m.effect_sizes_.set_index('feature')
    assert list(es.index) == ['a', 'b', 'c']
    # shares of the total, intervals, null reference
    assert es['effect_share'].sum() == pytest.approx(100.0)
    assert (es['effect_ci_low'] < es['effect']).all()
    assert (es['effect'] < es['effect_ci_high']).all()
    assert es.loc['a', 'effect'] > es.loc['a', 'effect_shuffled']
    assert es.loc['a', 'effect'] > es.loc['b', 'effect']
    assert es['effect_rel_target_sd'].tolist() == pytest.approx(
        (es['effect'] / m.target_std_).tolist())
    # y = 2 a + c + noise: positive direction of a, about +2 per SD of a
    assert es.loc['a', 'corr_x_shap'] > 0.8
    assert es.loc['a', 'slope_per_sd'] == pytest.approx(2.0, abs=0.6)
    # target-encoded predictor: direction undefined
    assert np.isnan(es.loc['c', 'slope_per_sd'])
    assert 'class' in m.effect_sizes_.columns


def test_effect_sizes_single_split_has_no_interval(data):
    m = _regressor(evaluation='TT').fit(data, data['y'], groups=data['pid'])
    es = m.effect_sizes_
    assert es['effect'].notna().all()
    assert es['effect_ci_low'].isna().all()
    assert es['slope_ci_low'].isna().all()
    summ = m.scores_summary_
    assert summ['p_value'].isna().all() and summ['ci_low'].isna().all()


def test_output_dir_layout(fitted):
    d, _ = fitted
    assert sorted(p.name for p in d.iterdir()) == [
        'final_model.joblib', 'folds', 'manifest.json']
    assert len(list((d / 'folds').glob('fold_*.json'))) == 5
    manifest = json.loads((d / 'manifest.json').read_text())
    assert manifest['complete'] is True and manifest['n_folds'] == 5
    assert manifest['final_model'] is True
    assert manifest['estimator'] == 'NCVGBDTRegressor'
    assert manifest['versions']['python'] and manifest['versions']['shap']
    # human readable
    assert manifest['params']['x_con_names'] == ['a', 'b']


def test_load_restores_everything_exactly(fitted, data):
    d, m = fitted
    loaded = load_model(d)
    assert isinstance(loaded, NCVGBDTRegressor)
    assert NCVGBDTRegressor.load(d).score_ == m.score_
    pd.testing.assert_frame_equal(loaded.cv_results_, m.cv_results_)
    pd.testing.assert_frame_equal(loaded.shap_effects_, m.shap_effects_)
    pd.testing.assert_frame_equal(loaded.scores_summary_, m.scores_summary_)
    pd.testing.assert_frame_equal(loaded.effect_sizes_, m.effect_sizes_)
    assert loaded.target_std_ == m.target_std_
    for a, b in zip(m.explanations_, loaded.explanations_):
        np.testing.assert_array_equal(a.values, b.values)
        np.testing.assert_array_equal(a.data, b.data)
        assert a.feature_names == b.feature_names
    assert list(loaded.feature_names_in_) == list(m.feature_names_in_)
    np.testing.assert_array_equal(loaded.used_index_, m.used_index_)
    assert loaded.target_name_ == 'y'
    # final model works after loading
    np.testing.assert_allclose(loaded.predict(data.head(10)),
                               m.predict(data.head(10)))
    # figures from the loaded estimator and straight from the directory
    out = make_figures(d, figures_dir=d / 'figs')
    assert len(list(Path(out).glob('*.png'))) >= 8


def test_load_wrong_class_and_missing_dir(fitted, tmp_path):
    d, _ = fitted
    with pytest.raises(TypeError, match='not a NCVGBDTClassifier'):
        NCVGBDTClassifier.load(d)
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / 'nothing')


def test_save_after_fit_without_output_dir(data, tmp_path):
    m = _regressor().fit(data, data['y'])
    with pytest.raises(ValueError, match='output_dir'):
        m.save()
    m.save(tmp_path / 'saved')
    loaded = load_model(tmp_path / 'saved')
    pd.testing.assert_frame_equal(loaded.cv_results_, m.cv_results_)


def test_version_mismatch_warns(fitted, tmp_path):
    d, _ = fitted
    copy = tmp_path / 'copy'
    shutil.copytree(d, copy)
    manifest = json.loads((copy / 'manifest.json').read_text())
    manifest['versions']['numpy'] = '0.0.1'
    (copy / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.warns(UserWarning, match='versions differ'):
        load_model(copy)


# --- refit / final model -------------------------------------------------------------

def test_final_model_predict_and_explain(fitted, data):
    _, m = fitted
    pred = m.predict(data.head(20))
    assert pred.shape == (20,)
    # strongly related to the truth although it is trained on the same rows
    assert np.corrcoef(pred, data['y'].head(20))[0, 1] > 0.8
    exp = m.explain(data.head(12))
    assert exp.values.shape == (12, 3)
    assert exp.feature_names == ['a', 'b', 'c']
    assert m.explain(data, max_samples=7).values.shape[0] == 7
    # prediction is permutation invariant w.r.t. extra columns / order
    shuffled = data[['pid', 'c', 'b', 'a', 'y']].head(20)
    np.testing.assert_allclose(m.predict(shuffled), pred)


def test_classifier_refit_with_string_labels(data, tmp_path):
    m = NCVGBDTClassifier(x_con_names=['a', 'b'], refit=True,
                          output_dir=tmp_path, **FAST)
    m.fit(data, data['bin'], groups=data['pid'])
    assert list(m.classes_) == ['no', 'yes']
    proba = m.predict_proba(data.head(10))
    assert proba.shape == (10, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(m.predict(data.head(10))) <= {'no', 'yes'}
    cv = m.cv_results_
    for col in ('acc', 'auc', 'logloss', 'brier', 'mcc', 'kappa', 'f1_macro',
                'mcfadden', 'cox_snell', 'nagelkerke', 'tjur'):
        assert col in cv.columns
    assert cv['auc'].between(0, 1).all()
    assert m.scores_summary_.loc['auc', 'mean'] > 0.5
    es = m.effect_sizes_
    assert set(es['class']) == {'yes'} and 'effect_rel_target_sd' not in es
    pd.testing.assert_frame_equal(load_model(tmp_path).effect_sizes_, es)
    # binary effects are reported for the positive class only
    assert set(m.shap_effects_['class']) == {'yes'}
    loaded = load_model(tmp_path)
    assert list(loaded.classes_) == ['no', 'yes']
    np.testing.assert_allclose(loaded.predict_proba(data.head(10)), proba)
    np.testing.assert_array_equal(loaded.results_['scores'][0]['y_true'],
                                  m.results_['scores'][0]['y_true'])
    assert loaded.results_['scores'][0]['class_weights'] == \
        m.results_['scores'][0]['class_weights']


# --- resume -----------------------------------------------------------------------------

def test_resume_reproduces_full_run(fitted, data, tmp_path):
    d, m = fitted
    copy = tmp_path / 'partial'
    shutil.copytree(d, copy)
    # simulate an interrupted run: only 3 of 5 folds were written
    for f in sorted((copy / 'folds').glob('fold_*.json'))[3:]:
        f.unlink()
    manifest = json.loads((copy / 'manifest.json').read_text())
    manifest['complete'] = False
    (copy / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.warns(UserWarning, match='incomplete'):
        partial = load_model(copy)
    assert len(partial.cv_results_) == 3
    resumed = _regressor(copy, refit=True, resume=True).fit(
        data, data['y'], groups=data['pid'])
    pd.testing.assert_frame_equal(resumed.cv_results_, m.cv_results_)
    assert len(list((copy / 'folds').glob('fold_*.json'))) == 5
    assert json.loads((copy / 'manifest.json').read_text())['complete']


def test_resume_refuses_different_settings_or_data(fitted, data, tmp_path):
    d, _ = fitted
    copy = tmp_path / 'copy'
    shutil.copytree(d, copy)
    with pytest.raises(ValueError, match='different data or settings'):
        _regressor(copy, resume=True, seed=1).fit(
            data, data['y'], groups=data['pid'])
    with pytest.raises(ValueError, match='different data or settings'):
        _regressor(copy, resume=True).fit(
            data, data['y'] * 2, groups=data['pid'])


def test_refit_without_resume_overwrites(fitted, data, tmp_path):
    d, m = fitted
    copy = tmp_path / 'copy'
    shutil.copytree(d, copy)
    m2 = _regressor(copy, seed=5).fit(data, data['y'], groups=data['pid'])
    manifest = json.loads((copy / 'manifest.json').read_text())
    assert manifest['params']['seed'] == 5
    assert not (copy / 'final_model.joblib').exists()  # old final model gone
    assert len(list((copy / 'folds').glob('fold_*.json'))) == 5
    assert not np.allclose(m2.cv_results_['score'], m.cv_results_['score'])


def test_no_unexpected_warnings_on_load(fitted):
    d, _ = fitted
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        load_model(d)
