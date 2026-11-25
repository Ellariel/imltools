"""Example: regression on the mtcars data set (a smoke test, 32 rows only).

Fits mpg ~ wt + vs + gear with repeated 5-fold cross-validation, stores the
results as JSON in ``./test/mpg`` and creates the figures. Run with
``uv run example.py``.
"""
import pandas as pd
import statsmodels.api as sm

from ncvgbdt import NCVGBDTRegressor, load_model, make_figures

if __name__ == '__main__':
    df = pd.DataFrame(sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data)

    model = NCVGBDTRegressor(x_con_names=['wt'],
                             x_cat_bin_names=['vs'],
                             x_cat_mult_names=['gear'],
                             evaluation='CV',
                             n_rep_outer_cv=1,
                             n_samples_inner_cv=10,
                             n_samples_rs=10,
                             max_samples_shap=10,
                             shap_interactions=False,
                             output_dir='./test/mpg')
    model.fit(df, df['mpg'])

    print(model.cv_results_[['fold', 'score', 'score_shuffled']])
    print(model.shap_effects_.groupby('feature')['effect'].mean())

    make_figures(model)

    # Later, e.g. in another session:
    model = load_model('./test/mpg')
    print('mean R2:', model.score_)
