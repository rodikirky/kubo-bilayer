import numpy as np

def kz_coverage_metrics(
    kz_diag: np.ndarray,
    amp_diag: np.ndarray,
    kz_fft: np.ndarray,
    *,
    p: float = 2.0,
    q_levels: tuple[float, ...] = (0.95, 0.99),
) -> dict[str, float]:
    """
    Quantify whether the FFT kz window is wide enough relative to a wide diagnostic kz scan.

    Context
    -------
    In this project, real-space bulk Green's functions G(Δz) are computed via an FFT over kz.
    The kz grid available to the FFT is determined by the real-space box length L=2*z_max:
        dk = 2π / L,  kz_fft spans a finite interval.
    To avoid wrap-around/aliasing artifacts, the FFT kz window should cover the kz-region
    where the k-space Green's function has significant weight.

    Inputs
    ------
    kz_diag, amp_diag:
        Wide diagnostic grid kz_diag (sorted increasing) and an amplitude profile amp_diag,
        typically amp_diag = |G_ij(kz_diag)| for a fixed matrix entry (i,j).
        This grid is independent of the FFT box and should be wide enough to capture peaks.
    kz_fft:
        The kz grid used internally by the FFT-based transform (usually narrower).

    Weighting
    ---------
    We define a nonnegative "weight density"
        w(kz) = |amp_diag(kz)|^p,
    with p >= 1 (default p=2). Larger p emphasizes sharp peaks more strongly.

    Metrics
    -------
    1) mass_fraction_inside_fft:
        Fraction of total weight on the diagnostic grid that lies inside the FFT kz interval
        [min(kz_fft), max(kz_fft)]:
            ∫_{FFT-range} w dkz / ∫_{diag-range} w dkz
        Values near 1 indicate good coverage; low values indicate truncation.

    2) K_q support radii and coverage ratios:
        For each q in q_levels (e.g. 0.95, 0.99), define K_q as the smallest radius K such that
            ∫_{-K..K} w dkz / ∫ w dkz >= q.
        Then define:
            coverage_ratio_q = max(|kz_fft|) / K_q
        Ratios >= 1 mean the FFT half-range covers at least q of the weighted mass.

    Notes on numerics
    -----------------
    - Integrals are approximated with trapezoidal quadrature on kz_diag.
    - To evaluate ∫_{a..b} efficiently for many candidate K, we build a prefix integral
      and use linear interpolation of that prefix to approximate integrals on sub-intervals.

    Returns
    -------
    A dict containing:
      - mass_fraction_inside_fft
      - kz_fft_min, kz_fft_max, kz_fft_max_abs
      - K_95, K_99, coverage_ratio_95, coverage_ratio_99 (depending on q_levels)
      - kz_peak and peak_inside_fft as simple additional alarms

    Interpretation (rule of thumb)
    ------------------------------
    - mass_fraction_inside_fft >= 0.95 and coverage_ratio_99 >= 1.0 are good signs.
    - If peaks lie outside FFT coverage, increase z_max (increases L, decreases dk, increases kz range),
      or adjust k_max / grid design depending on how you construct kz_fft in your codebase.
    """
    kz_diag = np.asarray(kz_diag)
    amp_diag = np.asarray(amp_diag)
    kz_fft = np.asarray(kz_fft)

    # assume kz_diag is sorted increasing (it should be)
    if kz_diag.ndim != 1 or amp_diag.ndim != 1 or kz_diag.size != amp_diag.size:
        raise ValueError("kz_diag and amp_diag must be 1D arrays of same length.")
    if not np.all(np.diff(kz_diag) >= 0):
        raise ValueError("kz_diag must be sorted increasing for these metrics.")

    w = np.power(np.abs(amp_diag), p)
    total = float(np.trapz(w, kz_diag))
    if total <= 0 or not np.isfinite(total):
        return {
            "mass_fraction_inside_fft": float("nan"),
            "kz_fft_min": float(np.min(kz_fft)),
            "kz_fft_max": float(np.max(kz_fft)),
            "kz_fft_max_abs": float(np.max(np.abs(kz_fft))),
        }

    kz_min = float(np.min(kz_fft))
    kz_max = float(np.max(kz_fft))
    kz_max_abs = float(np.max(np.abs(kz_fft)))

    inside = (kz_diag >= kz_min) & (kz_diag <= kz_max)
    mass_inside = float(np.trapz(w[inside], kz_diag[inside])) if np.any(inside) else 0.0
    mass_frac_inside = mass_inside / total

    # prefix integral I[i] = ∫_{kz[0]}^{kz[i]} w dkz
    I = np.zeros_like(kz_diag, dtype=float)
    dk = np.diff(kz_diag)
    I[1:] = np.cumsum(0.5 * (w[:-1] + w[1:]) * dk)

    def integral_between(a: float, b: float) -> float:
        # integral of w from a..b (clipped to domain), using linear interpolation on prefix
        a = float(np.clip(a, kz_diag[0], kz_diag[-1]))
        b = float(np.clip(b, kz_diag[0], kz_diag[-1]))
        if b <= a:
            return 0.0

        def I_at(x: float) -> float:
            j = int(np.searchsorted(kz_diag, x, side="right") - 1)
            j = max(0, min(j, kz_diag.size - 2))
            x0, x1 = kz_diag[j], kz_diag[j + 1]
            if x1 == x0:
                return float(I[j])
            t = (x - x0) / (x1 - x0)
            # linear interpolate prefix integral between nodes
            return float(I[j] + t * (I[j + 1] - I[j]))

        return I_at(b) - I_at(a)

    # find K_q by scanning candidate K values on the diag grid
    abs_k = np.unique(np.abs(kz_diag))
    abs_k.sort()

    out: dict[str, float] = {
        "mass_fraction_inside_fft": float(mass_frac_inside),
        "kz_fft_min": kz_min,
        "kz_fft_max": kz_max,
        "kz_fft_max_abs": kz_max_abs,
    }

    for q in q_levels:
        Kq = float("nan")
        for K in abs_k:
            frac = integral_between(-K, K) / total
            if frac >= q:
                Kq = float(K)
                break
        out[f"K_{int(q*100)}"] = Kq
        out[f"coverage_ratio_{int(q*100)}"] = kz_max_abs / Kq if np.isfinite(Kq) and Kq > 0 else float("nan")

    # quick extra: where is the max peak?
    k_peak = float(kz_diag[int(np.argmax(np.abs(amp_diag)))])
    out["kz_peak"] = k_peak
    out["peak_inside_fft"] = float((k_peak >= kz_min) and (k_peak <= kz_max))

    return out
