"""Interpretable machine learning toolbox: GBDT + nested CV + SHAP.

Typical use::

    from ncvgbdt import NCVGBDTRegressor, make_figures

    model = NCVGBDTRegressor(x_con_names=['wt'], output_dir='out')
    model.fit(df, df['mpg'], groups=None)
    print(model.cv_results_, model.shap_effects_)
    make_figures(model)

    # later
    from ncvgbdt import load_model
    model = load_model('out')

Estimators live in :mod:`ncvgbdt.estimator`, building blocks in
:mod:`ncvgbdt.fitting`, figures in :mod:`ncvgbdt.plotting` (number formatting
in :mod:`ncvgbdt.format`).
"""

from importlib.metadata import PackageNotFoundError, version

from .estimator import NCVGBDTClassifier
from .estimator import NCVGBDTRegressor
from .estimator import load_model
from .plotting import make_figures

try:
    __version__ = version('imltools')  # name of the installed distribution
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = 'unknown'

__all__ = ['NCVGBDTRegressor', 'NCVGBDTClassifier', 'load_model',
           'make_figures', '__version__']
