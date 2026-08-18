"""Group-max probe: propagate the ensemble score inside repeated content/owner groups.

The blend probability of every test row is mixed with the *maximum* blend probability of
its `id_content` group and of its `id_content_owner` group, both computed over the test
block only (no labels involved, so no leakage):

    p_adj = (1 - wc - wo) * p + wc * max_p_over_id_content + wo * max_p_over_id_content_owner

On the 4 chronological folds (group-max over the validation block, mirroring inference)
the equal4 ensemble improves from F1 0.4206 to 0.4232 at m=1.8 with wc=0.15, wo=0.10 and
is never worse on any fold. The probe keeps the shipped k=1782.

    python src/make_gmax_probe.py
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
WC, WO = 0.15, 0.10


def main() -> None:
    data = np.load(ROOT / "artifacts" / "probs_test.npz", allow_pickle=True)
    test = pd.read_csv(ROOT / "test.csv")
    assert (test["claim_id"].astype(str).to_numpy() == data["claim_id"].astype(str)).all()
    prob = np.mean(
        [np.mean([data[f"{c}_s{s}"] for s in SEEDS], axis=0) for c in COMPONENTS], axis=0
    )
    gmax_content = (
        pd.DataFrame({"p": prob, "k": test["id_content"].astype(str)})
        .groupby("k")["p"].transform("max").to_numpy()
    )
    gmax_owner = (
        pd.DataFrame({"p": prob, "k": test["id_content_owner"].astype(str)})
        .groupby("k")["p"].transform("max").to_numpy()
    )
    adjusted = (1 - WC - WO) * prob + WC * gmax_content + WO * gmax_owner
    order = np.argsort(-adjusted, kind="stable")
    labels = np.zeros(len(adjusted), dtype=int)
    labels[order[:K]] = 1
    out = pd.DataFrame({"claim_id": test["claim_id"], "is_valid": labels})
    path = ROOT / "submissions" / "submission_equal4_m180_gmax.csv"
    out.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path.name}: positives={labels.sum()} sha256={digest}")


if __name__ == "__main__":
    main()
