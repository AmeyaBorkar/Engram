"""Statistical helpers for retrieval / bench evaluation.

Two tests we hold every SOTA claim to:

  1. Bootstrap 95% confidence interval on per-question metrics.
     Used to decide if a config's mean recall@10 is meaningfully
     different from baseline (CI on the lift excludes 0).
  2. McNemar's exact test for paired pass/fail comparisons (e.g.,
     hit@10 between two configs on the same N questions).

No external dependencies beyond stdlib + numpy (already in the
project tree via sentence-transformers).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class _BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    n_samples: int


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_iters: int = 10000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> _BootstrapResult:
    """Bootstrap CI on the mean of `values`.

    Resamples `values` with replacement `n_iters` times; reports the
    [alpha/2, 1-alpha/2] percentile of the resampled means. Empty input
    -> (0, 0, 0). Single-element input -> (v, v, v) (no variance to
    estimate).
    """
    n = len(values)
    if n == 0:
        return _BootstrapResult(mean=0.0, ci_low=0.0, ci_high=0.0, n_samples=0)
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if n == 1:
        return _BootstrapResult(mean=mean, ci_low=mean, ci_high=mean, n_samples=1)
    rng = np.random.default_rng(seed)
    # Pre-sample indices in one shot for speed.
    idx = rng.integers(0, n, size=(n_iters, n))
    resamples = arr[idx].mean(axis=1)
    lo = float(np.quantile(resamples, alpha / 2))
    hi = float(np.quantile(resamples, 1 - alpha / 2))
    return _BootstrapResult(mean=mean, ci_low=lo, ci_high=hi, n_samples=n)


@dataclass(frozen=True, slots=True)
class _DiffCIResult:
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    excludes_zero: bool
    n_samples: int


def bootstrap_paired_diff_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    n_iters: int = 10000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> _DiffCIResult:
    """Bootstrap CI on the mean of (b - a) across paired observations.

    Assumes values_a[i] and values_b[i] are measurements of the same
    question under two configs. Resamples question indices (NOT
    independently) so the pairing is preserved. The lift CI is the
    statistical "did this feature help?" answer.
    """
    n = len(values_a)
    if n != len(values_b):
        raise ValueError(
            f"values_a and values_b must be same length: {n} != {len(values_b)}"
        )
    if n == 0:
        return _DiffCIResult(
            mean_a=0.0, mean_b=0.0, diff=0.0, ci_low=0.0, ci_high=0.0,
            excludes_zero=False, n_samples=0,
        )
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    diff_per_q = b - a
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    diff = float(diff_per_q.mean())
    if n == 1:
        return _DiffCIResult(
            mean_a=mean_a, mean_b=mean_b, diff=diff,
            ci_low=diff, ci_high=diff, excludes_zero=False, n_samples=1,
        )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iters, n))
    resamples = diff_per_q[idx].mean(axis=1)
    lo = float(np.quantile(resamples, alpha / 2))
    hi = float(np.quantile(resamples, 1 - alpha / 2))
    excludes = (lo > 0.0) or (hi < 0.0)
    return _DiffCIResult(
        mean_a=mean_a, mean_b=mean_b, diff=diff,
        ci_low=lo, ci_high=hi, excludes_zero=excludes, n_samples=n,
    )


@dataclass(frozen=True, slots=True)
class _McNemarResult:
    n_passes_only_a: int  # baseline passes, candidate fails
    n_passes_only_b: int  # candidate passes, baseline fails
    n_both_pass: int
    n_both_fail: int
    p_value: float
    test: str  # "exact-binomial" or "chi2"


def mcnemar(
    passes_a: Sequence[bool] | Sequence[int],
    passes_b: Sequence[bool] | Sequence[int],
) -> _McNemarResult:
    """McNemar's test for paired binary outcomes.

    For small discordant pair counts (< 25) uses the exact binomial
    test; otherwise the Yates-corrected chi-square approximation.
    Inputs are paired pass-arrays of equal length over the same
    questions.
    """
    n = len(passes_a)
    if n != len(passes_b):
        raise ValueError(
            f"passes_a and passes_b must be same length: {n} != {len(passes_b)}"
        )
    only_a = 0  # baseline pass, candidate fail
    only_b = 0  # candidate pass, baseline fail
    both_pass = 0
    both_fail = 0
    for a, b in zip(passes_a, passes_b, strict=True):
        a_bool = bool(a)
        b_bool = bool(b)
        if a_bool and b_bool:
            both_pass += 1
        elif a_bool and not b_bool:
            only_a += 1
        elif (not a_bool) and b_bool:
            only_b += 1
        else:
            both_fail += 1
    disc = only_a + only_b
    if disc == 0:
        return _McNemarResult(
            n_passes_only_a=only_a, n_passes_only_b=only_b,
            n_both_pass=both_pass, n_both_fail=both_fail,
            p_value=1.0, test="degenerate",
        )
    if disc < 25:
        # Exact binomial two-sided test: under H0, only_b ~ Binomial(disc, 0.5).
        k = min(only_a, only_b)
        # Two-sided p = 2 * P(X <= k) capped at 1.
        p = _binomial_cdf(k, disc, 0.5) * 2.0
        return _McNemarResult(
            n_passes_only_a=only_a, n_passes_only_b=only_b,
            n_both_pass=both_pass, n_both_fail=both_fail,
            p_value=min(p, 1.0), test="exact-binomial",
        )
    # Yates-corrected chi-square.
    stat = ((abs(only_a - only_b) - 1) ** 2) / disc
    p = _chi2_sf_df1(stat)
    return _McNemarResult(
        n_passes_only_a=only_a, n_passes_only_b=only_b,
        n_both_pass=both_pass, n_both_fail=both_fail,
        p_value=p, test="chi2",
    )


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p). Stable log-space accumulation."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    log_p = math.log(p) if p > 0 else float("-inf")
    log_q = math.log(1 - p) if p < 1 else float("-inf")
    log_terms = []
    for i in range(k + 1):
        log_term = (
            _log_comb(n, i)
            + (i * log_p if i > 0 else 0.0)
            + ((n - i) * log_q if (n - i) > 0 else 0.0)
        )
        log_terms.append(log_term)
    # logsumexp
    m = max(log_terms)
    s = sum(math.exp(t - m) for t in log_terms)
    return math.exp(m) * s


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _chi2_sf_df1(x: float) -> float:
    """Survival function (= 1 - CDF) of chi-square with 1 df.

    For df=1, chi2 = Z^2, so P(chi2 > x) = 2 * (1 - Phi(sqrt(x))).
    Uses an Abramowitz approximation for the normal CDF -- accurate to
    ~7 decimals, fine for p-value reporting.
    """
    if x <= 0:
        return 1.0
    z = math.sqrt(x)
    # 1 - Phi(z) via complementary error function
    return math.erfc(z / math.sqrt(2.0))


def format_ci(mean: float, lo: float, hi: float) -> str:
    """Pretty-print a CI as `mean [lo, hi]` with 3 decimals."""
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def format_p(p: float) -> str:
    """Pretty-print a p-value with scientific notation when small."""
    if p < 1e-4:
        return f"p<{1e-4:.0e}"
    return f"p={p:.4f}"
