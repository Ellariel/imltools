"""Tests of the figure interface: make_figures returns (fig, ax) per figure
name, selects figures with plot_type, applies figsize / plot_kwargs, and only
writes files when asked to."""
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ncvgbdt import (  # noqa: E402
    NCVGBDTClassifier, NCVGBDTRegressor, make_figures)
from ncvgbdt.plotting import _call, save_figures  # noqa: E402

FAST = dict(n_rep_outer_cv=1, n_samples_inner_cv=10, n_samples_rs=4,
            max_samples_shap=15)
DEFAULT_KEYS = {"fit_scatter", "fit_violin", "shap_effects",
                "shap_effects_distribution", "shap_values"}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({"pid": np.repeat(np.arange(40), 3),
                       "a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = df.a * 2 + rng.normal(size=n)
    df["bin"] = (df.a + rng.normal(size=n) > 0).astype(int)
    return df


@pytest.fixture(scope="module")
def reg(data):
    """Regressor with interaction values (all figure types available)."""
    return NCVGBDTRegressor(x_con_names=["a", "b"], shap_interactions=True,
                            **FAST).fit(data, "y", groups="pid")


@pytest.fixture(scope="module")
def reg_plain(data):
    return NCVGBDTRegressor(x_con_names=["a", "b"], shap_interactions=False,
                            **FAST).fit(data, "y", groups="pid")


@pytest.fixture(scope="module")
def clf(data):
    return NCVGBDTClassifier(x_con_names=["a", "b"], shap_interactions=False,
                             **FAST).fit(data, "bin", groups="pid")


def _is_axes(ax):
    items = np.ravel(ax) if isinstance(ax, np.ndarray) else [ax]
    return all(isinstance(a, Axes) for a in items)


def test_returns_named_fig_ax_pairs_and_writes_nothing(reg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a stray file would end up here
    figs = make_figures(reg)
    assert set(figs) == DEFAULT_KEYS
    for name, (fig, ax) in figs.items():
        assert isinstance(fig, Figure) and _is_axes(ax), name
    assert list(tmp_path.iterdir()) == []


def test_every_figure_is_its_own_figure(reg):
    """Figures stay open; SHAP and pandas draw into the current figure, so
    each one must start on a fresh figure and not overwrite an earlier one."""
    before = len(plt.get_fignums())
    figs = make_figures(reg, plot_type="all")
    assert len(plt.get_fignums()) - before == len(figs)
    numbers = [fig.number for fig, _ in figs.values()]
    assert len(set(numbers)) == len(numbers)
    # the bar plot holds one bar per predictor and nothing else
    fig, ax = figs["shap_effects"]
    assert len(ax.patches) == 2 and len(fig.axes) == 1
    # the beeswarm figure holds its own axes plus the colorbar
    fig, ax = figs["shap_values"]
    assert len(fig.axes) == 2


def test_plot_type_selection(reg, reg_plain):
    assert set(make_figures(reg, plot_type="fit")) == {"fit_scatter", "fit_violin"}
    assert set(make_figures(reg, plot_type=["shap_effects", "fit"])) == {
        "fit_scatter", "fit_violin", "shap_effects"}
    # fixed drawing order, whatever the order asked for
    assert list(make_figures(reg, plot_type=["shap_values", "fit"]))[:2] == [
        "fit_scatter", "fit_violin"]
    everything = make_figures(reg, plot_type="all")
    assert {"parameters:estimator__regressor__path_smooth", "dependence:a",
            "dependence:b", "interaction_effects",
            "interaction_values:a:b"} <= set(everything)
    # 'all' skips what does not apply instead of failing
    plain = make_figures(reg_plain, plot_type="all")
    assert not any(k.startswith("interaction") for k in plain)
    assert "dependence:a" in plain


def test_plot_type_errors(reg, reg_plain):
    with pytest.raises(ValueError, match="Unknown plot_type"):
        make_figures(reg, plot_type="nope")
    with pytest.raises(ValueError, match="Unknown plot_type"):
        make_figures(reg, plot_type=["fit", "nope"])
    with pytest.raises(ValueError, match="shap_interactions=True"):
        make_figures(reg_plain, plot_type="interaction_values")
    # nothing is drawn when the arguments are wrong
    before = len(plt.get_fignums())
    with pytest.raises(ValueError):
        make_figures(reg, plot_type=["fit", "nope"])
    assert len(plt.get_fignums()) == before


def test_figsize(reg):
    one = make_figures(reg, plot_type=["fit", "shap_effects"], figsize=(5, 3))
    for fig, _ in one.values():
        assert tuple(fig.get_size_inches()) == (5, 3)
    some = make_figures(reg, plot_type=["fit", "shap_effects"],
                        figsize={"fit": (6, 2)})
    assert tuple(some["fit_scatter"][0].get_size_inches()) == (6, 2)
    default = make_figures(reg, plot_type="shap_effects")
    assert tuple(some["shap_effects"][0].get_size_inches()) == tuple(
        default["shap_effects"][0].get_size_inches())
    for bad in ((5,), (0, 3), "big", {"nope": (5, 3)}, {"fit": 5}):
        with pytest.raises(ValueError, match="figsize"):
            make_figures(reg, figsize=bad)


def test_plot_kwargs(reg):
    figs = make_figures(reg, plot_type="fit",
                        plot_kwargs={"fit_scatter": {"color": "red", "alpha": 0.5}})
    points = figs["fit_scatter"][1].collections[0]
    assert tuple(points.get_facecolor()[0][:3]) == to_rgba("red")[:3]
    for bad in ({"nope": {}}, {"fit_scatter": "red"}, {"fit": {}}):
        with pytest.raises(ValueError, match="plot_kwargs"):
            make_figures(reg, plot_type="fit", plot_kwargs=bad)


def test_call_lets_user_arguments_override_defaults():
    seen = {}

    def fn(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)

    _call(fn, {"color": "red"}, 1, 2, color="blue", alpha=0.5)
    assert seen == {"args": (1, 2), "kwargs": {"color": "red", "alpha": 0.5}}


def test_classes_selection(clf, reg):
    both = make_figures(clf, plot_type="shap_effects")
    assert set(both) == {"shap_effects:class_0", "shap_effects:class_1"}
    one = make_figures(clf, plot_type="shap_effects", classes=[1])
    assert set(one) == {"shap_effects:class_1"}
    with pytest.raises(ValueError, match="Unknown classes"):
        make_figures(clf, classes=[7])
    with pytest.raises(ValueError, match="classification"):
        make_figures(reg, classes=[1])
    assert {"fit_confusion", "fit_violin"} <= set(make_figures(clf, plot_type="fit"))


@pytest.mark.parametrize("correction, column", [
    (None, "p_value"), (False, "p_value"), (True, "p_value_bonferroni"),
    ("bonferroni", "p_value_bonferroni"), ("holm", "p_value_holm"),
    ("fdr_bh", "p_value_fdr_bh")])
def test_effect_figure_shows_the_table_p_values(reg_plain, correction, column):
    """The p values in the figure are those of effect_sizes_."""
    from ncvgbdt.format import format_p
    figs = make_figures(reg_plain, plot_type="shap_effects",
                        multiple_comparison_correction=correction)
    shown = sorted(t.get_text().split("|")[1] for t in figs["shap_effects"][1].texts)
    expected = sorted(format_p(p) for p in reg_plain.effect_sizes_[column])
    assert shown == expected


def test_interaction_figure_shows_the_table_p_values(reg):
    figs = make_figures(reg, plot_type="interaction_effects",
                        multiple_comparison_correction="holm")
    shown = sorted(t.get_text().split("\n")[1] for t in figs["interaction_effects"][1].texts)
    expected = sorted(f"p {round(p, 3)}" for p in reg.interaction_effects_["p_value_holm"])
    assert shown == expected


def test_correction_argument_is_validated(reg_plain):
    for bad in ("nope", 2, "Holm"):
        with pytest.raises(ValueError, match="multiple_comparison_correction"):
            make_figures(reg_plain, multiple_comparison_correction=bad)


def test_shap_values_style_and_correction(reg_plain):
    graph = make_figures(reg_plain, plot_type="shap_values",
                         shap_values_style="graph",
                         multiple_comparison_correction=True)
    assert set(graph) == {"shap_values"}
    with pytest.raises(ValueError, match="shap_values_style"):
        make_figures(reg_plain, shap_values_style="nope")


def test_saving_formats_and_dpi(reg, tmp_path):
    out = tmp_path / "new" / "dir"            # created on demand
    figs = make_figures(reg, plot_type="fit", output_dir=out,
                        fig_format=["png", "svg", "pdf"], dpi=50)
    assert sorted(p.name for p in out.iterdir()) == sorted(
        f"{n}.{ext}" for n in figs for ext in ("png", "svg", "pdf"))
    low = plt.imread(out / "fit_scatter.png")
    save_figures(figs, tmp_path / "hi", "png", dpi=100)
    high = plt.imread(tmp_path / "hi" / "fit_scatter.png")
    assert abs(high.shape[0] / low.shape[0] - 2) < 0.05    # dpi sets the pixels
    # no files for a single format request other than the ones asked for
    only = make_figures(reg, plot_type="fit", output_dir=tmp_path / "one")
    assert {p.suffix for p in (tmp_path / "one").iterdir()} == {".png"}
    assert len(only) == 2


def test_saving_errors(reg, tmp_path):
    for bad in ("jpg", ["png", "tiff"], []):
        with pytest.raises(ValueError, match="fig_format"):
            make_figures(reg, plot_type="fit", output_dir=tmp_path, fig_format=bad)
    for bad in (0, -5, "high", True):
        with pytest.raises(ValueError, match="dpi"):
            make_figures(reg, plot_type="fit", output_dir=tmp_path, dpi=bad)
    assert list(tmp_path.iterdir()) == []


def test_edit_then_save(reg, tmp_path):
    figs = make_figures(reg, plot_type="shap_effects")
    fig, ax = figs["shap_effects"]
    ax.set_title("edited title")
    paths = save_figures(figs, tmp_path, ["png", "pdf"], dpi=80)
    assert sorted(p.rsplit(".", 1)[-1] for p in paths) == ["pdf", "png"]
    assert all(os.path.exists(p) for p in paths)
    assert ax.get_title() == "edited title"


def test_save_figures_keeps_names_apart(tmp_path):
    """Names that become the same file name must not overwrite each other."""
    fig_a, ax_a = plt.subplots()
    fig_b, ax_b = plt.subplots()
    paths = save_figures({"a:b": (fig_a, ax_a), "a__b": (fig_b, ax_b)}, tmp_path)
    assert len(set(paths)) == 2 and len(list(tmp_path.iterdir())) == 2
