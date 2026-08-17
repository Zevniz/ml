"""Round-3 features: reporter fingerprint and backward-looking entity history.

Two ideas that the earlier rounds did not cover.

1. **Reporter fingerprint.** The data has no reporter id, and the earlier catalogue closed
   the "reporter history" direction as *unavailable*. A reporter can still be identified
   approximately by the tuple of its stable profile attributes (registration year, age
   bucket, sex, avatar/school/university/private flags, three country ids). That tuple has
   5,640 distinct values on train, 46,623 of 48,658 rows sit in a group of size > 1, and a
   past-only target encoding of the tuple alone reaches ROC AUC 0.60-0.64 on later rows, so
   it carries information that single columns do not (it is a 10-way interaction).

2. **Backward-looking entity history.** The existing pipeline has total complaint counts per
   content / owner (computed transductively on train+test), but nothing about the *order*
   and *spacing* of complaints. `content_prior_claims`, `owner_prior_claims`,
   `fp_prior_claims`, their pair counterparts and the gaps to the previous complaint are all
   computed on train+test **without labels** and only look backwards in time, so they are
   available for test rows exactly as they are for train rows.

Leakage rules kept identical to the rest of the pipeline: label-based encodings are
out-of-fold inside the training prefix and mapped onto validation/test from the prefix only;
non-label counters may use train+test (documented transductive step).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solution import REPORTER_FP_COLS as FP_COLS  # noqa: E402

TE_ALPHA = 15.0
N_SPLITS = 5
RANDOM_SEED = 42
NO_PREVIOUS = -1.0  # sentinel for "no earlier complaint with this key"


def fingerprint(df: pd.DataFrame) -> pd.Series:
    """Approximate reporter identity: join of stable profile attributes."""
    return df[FP_COLS].astype(str).agg("|".join, axis=1)


def _prior_stats(keys: pd.Series, seconds: np.ndarray, prefix: str) -> pd.DataFrame:
    """Number of earlier rows with the same key and the gap to the previous one.

    `keys` and `seconds` must already be ordered by event time.
    """
    frame = pd.DataFrame({"key": keys.to_numpy(), "t": seconds})
    grouped = frame.groupby("key", sort=False)
    prior = grouped.cumcount().to_numpy().astype(float)
    gap = grouped["t"].diff().to_numpy()
    span = (frame["t"] - grouped["t"].transform("first")).to_numpy()
    return pd.DataFrame(
        {
            f"{prefix}_prior_claims": prior,
            f"{prefix}_gap_prev_hours": np.where(np.isnan(gap), NO_PREVIOUS, gap / 3600.0),
            f"{prefix}_span_first_hours": np.where(prior == 0, NO_PREVIOUS, span / 3600.0),
        }
    )


def transductive_history(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking history features for every row of train+test, keyed by claim_id.

    No target column is touched, so computing this on the concatenation of train and test
    introduces no label leakage; each row only sees complaints that happened before it.
    """
    combined = pd.concat(
        [train.drop(columns=[c for c in ("is_valid",) if c in train.columns]), test],
        axis=0,
        ignore_index=True,
    )
    combined["event_dt"] = pd.to_datetime(combined["first_event_time"])
    combined = combined.sort_values("event_dt", kind="mergesort").reset_index(drop=True)
    seconds = combined["event_dt"].astype("int64").to_numpy() / 1e9

    fp = fingerprint(combined)
    keys = {
        "content": combined["id_content"].astype(str),
        "owner": combined["id_content_owner"].astype(str),
        "fp": fp,
        "fp_owner": fp + "||" + combined["id_content_owner"].astype(str),
        "fp_type": fp + "||" + combined["claim_type"].astype(str),
    }
    parts = [_prior_stats(key, seconds, name) for name, key in keys.items()]

    out = pd.concat(parts, axis=1)
    # transductive totals for the fingerprint mirror the existing content/owner counters
    fp_total = fp.map(fp.value_counts()).to_numpy().astype(float)
    out["fp_total_claims"] = fp_total
    out["fp_share_of_owner"] = out["fp_owner_prior_claims"] / np.maximum(out["fp_prior_claims"], 1.0)
    out["content_prior_distinct_fp"] = (
        pd.DataFrame({"c": combined["id_content"].astype(str), "f": fp})
        .groupby("c", sort=False)["f"]
        .transform(lambda s: (~s.duplicated()).cumsum() - 1)
        .to_numpy()
        .astype(float)
    )
    out.insert(0, "claim_id", combined["claim_id"].to_numpy())
    return out.set_index("claim_id")


def oof_target_encoding(
    train_keys: pd.Series,
    y: np.ndarray,
    other_keys: pd.Series,
    alpha: float = TE_ALPHA,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """K-fold OOF target encoding on the training prefix, mapped onto `other_keys`.

    Returns (te_train, te_other, count_train, count_other). Validation/test rows receive the
    encoding computed on the whole training prefix; unseen keys fall back to the prefix mean.
    """
    train_keys = train_keys.reset_index(drop=True)
    other_keys = other_keys.reset_index(drop=True)
    global_mean = float(np.mean(y))
    oof = np.full(len(train_keys), global_mean, dtype=float)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    frame = pd.DataFrame({"key": train_keys, "y": y})
    for fit_idx, hold_idx in kfold.split(frame):
        stats = frame.iloc[fit_idx].groupby("key")["y"].agg(["sum", "count"])
        mapping = ((stats["sum"] + alpha * global_mean) / (stats["count"] + alpha)).to_dict()
        oof[hold_idx] = frame.iloc[hold_idx]["key"].map(mapping).fillna(global_mean).to_numpy()
    stats_full = frame.groupby("key")["y"].agg(["sum", "count"])
    mapping_full = ((stats_full["sum"] + alpha * global_mean) / (stats_full["count"] + alpha)).to_dict()
    te_other = other_keys.map(mapping_full).fillna(global_mean).to_numpy()
    counts = stats_full["count"].to_dict()
    count_train = train_keys.map(counts).fillna(0).to_numpy().astype(float)
    count_other = other_keys.map(counts).fillna(0).to_numpy().astype(float)
    return oof, te_other, count_train, count_other
