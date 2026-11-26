"""Regression test based on the README quick start (mtcars, regression,
repeated CV).

Checks that (1) the analysis is reproducible for a fixed seed, (2) it matches
reference values, and (3) figures are produced.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from ncvgbdt import NCVGBDTRegressor, make_figures  # noqa: E402

SEED = 3141592

# mtcars (Motor Trend, 1974; as distributed with R): 32 rows, kept as a CSV
# file so that no extra package is needed to load it.
MTCARS = Path(__file__).parent / "data" / "mtcars.csv"

# Reference values produced with seed=3141592: R2 per outer fold and mean
# absolute SHAP value per feature of the first fold. Library versions change
# random number handling slightly, so there is one set per NumPy major version
# (numpy 1: numpy 1.26.4 / shap 0.46.0 / scikit-learn 1.5.2 / lightgbm 4.5.0,
#  numpy 2: current releases). Python 3.12 and 3.13 give identical numbers.
REFERENCES = {
    1: {
        "r2": [-5.025748220224649, 0.6178500488912753, 0.5376096658844915,
               0.8796622718882823, -0.017473123555114656],
        "shap0": [2.5754624427028405, 1.4330916401743703, 1.0300037812889298],
    },
    2: {
        "r2": [0.5744220628194414, 0.31965420046604076, 0.6168051993806891,
               0.6434127743766387, 0.9185376160070287],
        "shap0": [4.0335133035468, 1.071256766948107, 0.2031878098453746],
    },
}
NUMPY_MAJOR = int(np.__version__.split(".")[0])
# Tolerance for platform/BLAS level float differences (same library versions).
RTOL, ATOL = 1e-3, 1e-4


def _fit(results_dir=None, seed=SEED):
    df = pd.read_csv(MTCARS, index_col="model")
    model = NCVGBDTRegressor(x_con_names=['wt'],
                             x_cat_bin_names=['vs'],
                             x_cat_mult_names=['gear'],
                             n_rep_outer_cv=1,
                             n_samples_inner_cv=10,
                             n_samples_rs=10,
                             max_samples_shap=10,
                             shap_interactions=False,
                             seed=seed,
                             output_dir=results_dir)
    return model.fit(df, df['mpg'])


def _summary(model):
    res = model.results_
    r2 = [s['r2'] for s in res['scores']]
    r2_sh = [s['r2'] for s in res['scores_sh']]
    shap0 = np.abs(res['explainations'][0].values).mean(axis=0)
    return np.array(r2), np.array(r2_sh), shap0


@pytest.fixture(scope="module")
def run_a(tmp_path_factory):
    d = tmp_path_factory.mktemp("run_a")
    return d, _fit(d)


def _check_reference(model, ref):
    r2, _, shap0 = _summary(model)
    assert len(r2) == 5
    np.testing.assert_allclose(r2, ref["r2"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(shap0, ref["shap0"], rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(NUMPY_MAJOR != 1, reason="reference for numpy 1.x stack")
def test_expected_values_numpy1(run_a):
    _check_reference(run_a[1], REFERENCES[1])


@pytest.mark.skipif(NUMPY_MAJOR != 2, reason="reference for numpy 2.x stack")
def test_expected_values_numpy2(run_a):
    _check_reference(run_a[1], REFERENCES[2])


def test_reproducible(run_a):
    _, model_a = run_a
    model_b = _fit()
    for a, b in zip(_summary(model_a), _summary(model_b)):
        np.testing.assert_array_equal(a, b)
    assert model_a.results_['best_params'] == model_b.results_['best_params']


def test_figures(run_a):
    d, model = run_a
    figs = make_figures(model, output_dir=d / 'figs')
    assert {'fit_scatter', 'fit_violin', 'shap_effects',
            'shap_effects_distribution', 'shap_values'} <= set(figs)
    assert (d / 'figs' / 'fit_scatter.png').exists()
