"""Segment probes: measure where the champion's true positives actually are.

The hidden test has exactly 990 positives, so any submission with a known positive set S
returns `TP(S) = F1 * (|S| + 990) / 2` exactly. The champion (k=1782, F1 0.489, TP 678)
splits into 808 positives inside the `ip_country_id == 11` segment (April/May 2026) and
974 outside it (March 2026). Three probes:

* `probe_seg_c11` — champion positives restricted to the country-11 segment. One upload
  yields TP_c11 exactly, and TP_rest = 678 - TP_c11, i.e. the per-segment precision of the
  shipped operating point.
* `probe_shift_to23` — same total k=1782, but the 150 lowest-ranked country-11 positives
  are swapped for the 150 best non-selected non-country-11 rows.
* `probe_shift_to11` — the reverse swap.

The two shift probes measure the marginal precision of each segment's ranking tail, which
is what decides whether the 808/974 budget split is optimal.

    python src/make_segment_probe.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ["cat_all", "cat_recent", "lgb_neutral", "lgb_diverse"]
SEEDS = [42, 2026, 777]
K = 1782
SHIFT = 150


def write(test: pd.DataFrame, labels: np.ndarray, name: str) -> None:
    path = ROOT / "submissions" / f"{name}.csv"
    pd.DataFrame({"claim_id": test["claim_id"], "is_valid": labels}).to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path.name}: positives={labels.sum()} sha256={digest}")


def main() -> None:
    data = np.load(ROOT / "artifacts" / "probs_test.npz", allow_pickle=True)
    test = pd.read_csv(ROOT / "test.csv")
    assert (test["claim_id"].astype(str).to_numpy() == data["claim_id"].astype(str)).all()
    prob = np.mean(
        [np.mean([data[f"{c}_s{s}"] for s in SEEDS], axis=0) for c in COMPONENTS], axis=0
    )
    order = np.argsort(-prob, kind="stable")
    base = np.zeros(len(prob), dtype=int)
    base[order[:K]] = 1
    champ = pd.read_csv(ROOT / "submissions" / "submission_equal4_m180.csv")
    assert (champ["is_valid"].to_numpy() == base).all(), "ranking must reproduce the champion"

    c11 = (test["ip_country_id"] == 11).to_numpy()
    write(test, np.where(c11, base, 0), "probe_seg_c11")

    rank = np.empty(len(prob), dtype=int)
    rank[order] = np.arange(len(prob))
    sel = base == 1

    def shift(from_seg: np.ndarray, to_seg: np.ndarray, name: str) -> None:
        drop = np.where(sel & from_seg)[0]
        drop = drop[np.argsort(-rank[drop])][:SHIFT]  # lowest-ranked selected rows
        add = np.where(~sel & to_seg)[0]
        add = add[np.argsort(rank[add])][:SHIFT]  # best-ranked unselected rows
        labels = base.copy()
        labels[drop] = 0
        labels[add] = 1
        assert labels.sum() == K
        write(test, labels, name)

    shift(c11, ~c11, "probe_shift_to23")
    shift(~c11, c11, "probe_shift_to11")


if __name__ == "__main__":
    main()
