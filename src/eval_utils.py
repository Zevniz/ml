"""Shared evaluation helpers: top-k F1 on temporal folds and paired comparisons.

The hidden test contains exactly 990 positives (derived from the organiser score of the
all-ones submission), i.e. a prevalence of 990 / 7446. On a validation block of the same
size the comparable operating point is `k = m * prevalence * n`, so all offline
comparisons use the same rate rule as the submission.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TEST_POSITIVES = 990
BLOCK = 7446
TEST_PREVALENCE = TEST_POSITIVES / BLOCK
DEFAULT_M = 1.8


def f1_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    """F1 of the positive class when the top-`k` scored rows are predicted as 1."""
    k = int(min(max(k, 1), len(score)))
    order = np.argsort(-score, kind="stable")
    tp = float(y[order[:k]].sum())
    positives = float(y.sum())
    if tp == 0:
        return 0.0
    return 2.0 * tp / (k + positives)


def f1_at_m(y: np.ndarray, score: np.ndarray, m: float = DEFAULT_M) -> float:
    """Top-k F1 where k equals `m` times the number of positives implied by prevalence.

    The number of positives is taken from the known test prevalence scaled to the block
    length, never from `y`, so the rule is identical to the one used on the test set.
    """
    return f1_at_k(y, score, int(round(m * TEST_PREVALENCE * len(score))))


def fold_metrics(y: np.ndarray, score: np.ndarray, ms=(1.6, 1.8, 2.0)) -> dict:
    out = {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }
    for m in ms:
        out[f"f1_m{m:g}"] = f1_at_m(y, score, m)
    out["f1_oracle"] = max(f1_at_k(y, score, k) for k in range(200, 3500, 20))
    return out


def paired_bootstrap(y: np.ndarray, base: np.ndarray, cand: np.ndarray,
                     metric=f1_at_m, n_boot: int = 400, seed: int = 42) -> dict:
    """Paired bootstrap over rows: how often the candidate beats the baseline."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = metric(y[idx], cand[idx]) - metric(y[idx], base[idx])
    return {
        "mean_diff": float(diffs.mean()),
        "win_rate": float((diffs > 0).mean()),
        "p05": float(np.quantile(diffs, 0.05)),
        "p95": float(np.quantile(diffs, 0.95)),
    }
