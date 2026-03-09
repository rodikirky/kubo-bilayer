from __future__ import annotations

import numpy as np
import pytest

import kubo.plotting as plotting


# ---------------------------------------------
# region profile_amplitude_over_first_axis
# ---------------------------------------------
def test_profile_amplitude_vector_case_returns_abs():
    arr = np.array([1 + 2j, -3j, 4.0], dtype=complex)
    out = plotting.profile_amplitude_over_first_axis(arr)
    assert out.shape == (arr.size,)
    assert np.allclose(out, np.abs(arr))


def test_profile_amplitude_matrix_stack_fro_matches_slice_norm():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(7, 2, 2)) + 1j * rng.normal(size=(7, 2, 2))

    out = plotting.profile_amplitude_over_first_axis(arr, mode="fro")
    expected = np.array([np.linalg.norm(arr[i], ord="fro") for i in range(arr.shape[0])])

    assert out.shape == (7,)
    assert np.allclose(out, expected, atol=1e-12)


def test_profile_amplitude_matrix_stack_max_matches_max_abs_entry():
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(5, 3, 3)) + 1j * rng.normal(size=(5, 3, 3))

    out = plotting.profile_amplitude_over_first_axis(arr, mode="max")
    expected = np.max(np.abs(arr), axis=(1, 2))

    assert out.shape == (5,)
    assert np.allclose(out, expected, atol=0.0)


def test_profile_amplitude_elem00_matches_abs_00():
    rng = np.random.default_rng(2)
    arr = rng.normal(size=(9, 2, 2)) + 1j * rng.normal(size=(9, 2, 2))

    out = plotting.profile_amplitude_over_first_axis(arr, mode="elem00")
    expected = np.abs(arr[:, 0, 0])

    assert out.shape == (9,)
    assert np.allclose(out, expected, atol=0.0)


def test_profile_amplitude_rejects_invalid_ndim():
    with pytest.raises(ValueError):
        plotting.profile_amplitude_over_first_axis(np.zeros((2, 2), dtype=complex))


def test_profile_amplitude_rejects_unknown_mode():
    rng = np.random.default_rng(3)
    arr = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))

    with pytest.raises(ValueError):
        plotting.profile_amplitude_over_first_axis(arr, mode="nope")


# endregion
# ---------------------------------------------


# ---------------------------------------------
# region edge_leak_ratio
# ---------------------------------------------
def test_edge_leak_ratio_basic_expected_value():
    # center is 10, edges max is 1 => ratio 0.1
    p = np.array([1, 0, 0, 10, 0, 0, 1], dtype=float)
    r = plotting.edge_leak_ratio(p, m=2)  # edges are [1,0] and [0,1]
    assert r == pytest.approx(0.1)


def test_edge_leak_ratio_explicit_center_index():
    p = np.array([2, 2, 9, 2, 2], dtype=float)  # center at index 2 is 9
    r1 = plotting.edge_leak_ratio(p, m=1)  # edges [2] and [2] => 2/9
    r2 = plotting.edge_leak_ratio(p, m=1, center_index=2)
    assert r1 == pytest.approx(2 / 9)
    assert r2 == pytest.approx(2 / 9)


def test_edge_leak_ratio_center_zero_uses_small_denom():
    p = np.array([1.0, 0.0, 2.0], dtype=float)  # default center index = 1 => denom=0
    r = plotting.edge_leak_ratio(p, m=1)
    assert np.isfinite(r)
    assert r == pytest.approx(2.0 / 1e-300)


def test_edge_leak_ratio_rejects_non_1d_profile():
    with pytest.raises(ValueError):
        plotting.edge_leak_ratio(np.zeros((3, 1)), m=1)


@pytest.mark.parametrize("m", [0, -1])
def test_edge_leak_ratio_rejects_nonpositive_m(m: int):
    p = np.ones(5)
    with pytest.raises(ValueError):
        plotting.edge_leak_ratio(p, m=m)


def test_edge_leak_ratio_rejects_m_too_large():
    p = np.ones(5)
    with pytest.raises(ValueError):
        plotting.edge_leak_ratio(p, m=3)  # 2*m > size


def test_edge_leak_ratio_rejects_center_out_of_bounds():
    p = np.ones(7)
    with pytest.raises(ValueError):
        plotting.edge_leak_ratio(p, m=2, center_index=99)


# endregion
# ---------------------------------------------


# ---------------------------------------------
# region plotting wrappers (matplotlib optional)
# ---------------------------------------------
def _mpl_or_skip():
    mpl = pytest.importorskip("matplotlib")
    # try to avoid GUI backends in CI/headless runs
    try:
        mpl.use("Agg", force=True)
    except Exception:
        pass
    return mpl


def test_plot_profile_returns_fig_ax_and_sets_labels_and_scale():
    _mpl_or_skip()

    x = np.linspace(-1, 1, 11)
    y = np.exp(-x**2)

    fig, ax = plotting.plot_profile(
        x,
        y,
        title="MyTitle",
        xlabel="X",
        ylabel="Y",
        logy=True,
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_yscale() == "log"
    assert ax.get_title() == "MyTitle"
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"


def test_plot_profile_logy_false_keeps_linear_scale():
    _mpl_or_skip()

    x = np.arange(5)
    y = np.arange(5) + 1

    _, ax = plotting.plot_profile(
        x,
        y,
        title="t",
        xlabel="x",
        ylabel="y",
        logy=False,
    )

    assert ax.get_yscale() == "linear"


def test_plot_complex_components_plots_two_lines_with_legend():
    _mpl_or_skip()

    x = np.linspace(0, 1, 5)
    y = (1 + 2j) * x

    _, ax = plotting.plot_complex_components(
        x,
        y,
        title="c",
        xlabel="x",
        ylabel="y",
    )

    assert len(ax.lines) == 2
    labels = [line.get_label() for line in ax.lines]
    assert "Re" in labels
    assert "Im" in labels
    assert ax.get_legend() is not None


def test_show_calls_matplotlib_show(monkeypatch):
    _mpl_or_skip()

    called = {"n": 0}

    class DummyPlt:
        def show(self):
            called["n"] += 1

    monkeypatch.setattr(plotting, "_import_plt", lambda: DummyPlt())
    plotting.show()
    assert called["n"] == 1


# endregion
# ---------------------------------------------
