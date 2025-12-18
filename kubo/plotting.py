from __future__ import annotations

from typing import Optional

import numpy as np


def profile_amplitude_over_first_axis(arr: np.ndarray, *, mode: str = "fro") -> np.ndarray:
    """
    Turn a (N, dim, dim) complex matrix-valued array into a 1D amplitude profile of length N.

    mode:
      - "fro": Frobenius norm of the matrix at each index (default, robust)
      - "max": max absolute entry at each index (worst-case)
      - "elem00": |arr[:,0,0]| convenience (requires matrix)
    """
    arr = np.asarray(arr)

    if arr.ndim == 1: # (N,) vector case. Hence arr.ndim == 1
        return np.abs(arr)

    if arr.ndim != 3: # must be (N, dim, dim), i.e. a stack of matrices. Hence arr.ndim == 3.
        raise ValueError(f"Expected arr.ndim in {{1,3}}, got {arr.ndim} with shape {arr.shape}.")

    if mode == "fro": # Frobenius norm
        return np.array([np.linalg.norm(arr[i], ord="fro") for i in range(arr.shape[0])])
    if mode == "max": # max absolute magnitude entry
        return np.max(np.abs(arr), axis=(1, 2))
    if mode == "elem00": # (0,0) element absolute value
        return np.abs(arr[:, 0, 0])

    raise ValueError(f"Unknown mode={mode!r}.")


def edge_leak_ratio(profile: np.ndarray, *, m: int = 8, center_index: Optional[int] = None) -> float:
    """
    Ratio: max amplitude near edges / amplitude at center.
    Useful quick proxy for FFT wrap-around risk.

    Paraemeters
    ----------
    profile : np.ndarray
        1D array of amplitudes. 
    m : int
        Number of points from each edge to consider.
    center_index : Optional[int]
        Index of the center point. If None, uses profile.size // 2.
    
    Returns
    -------
    float
        The edge leak ratio.
    
    Raises
    ------
    ValueError
        If profile is not 1D, or m is negative or too large for profile size.
    """
    p = np.asarray(profile, dtype=float)
    if p.ndim != 1:
        raise ValueError("profile must be 1D.")
    if m <= 0:
        raise ValueError("m must be positive.")
    if 2 * m > p.size:
        raise ValueError("m too large for profile size.")

    c = (p.size // 2) if center_index is None else int(center_index)
    denom = p[c] if p[c] != 0.0 else 1e-300
    edges = np.r_[p[:m], p[-m:]]
    return float(edges.max() / denom)


def _import_plt():
    # Keep matplotlib optional: only import when plotting is actually called.
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install -e '.[plot]'"
        ) from e
    return plt


def plot_profile(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    logy: bool = True,
):
    plt = _import_plt()
    fig, ax = plt.subplots()
    ax.plot(x, y)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax


def plot_complex_components(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
):
    plt = _import_plt()
    fig, ax = plt.subplots()
    ax.plot(x, np.real(y), label="Re")
    ax.plot(x, np.imag(y), label="Im")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig, ax
