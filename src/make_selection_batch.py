"""Generate cached ranking-selection candidates at two operating points.

The script only reads cached component probabilities and never retrains models:

    python src/make_selection_batch.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exp_features import (  # noqa: E402
    BLEND_COMPONENTS,
    BLEND_SEEDS,
    _equal_component_score,
    _gmax_score,
    _rank_score,
)

K_MAIN = 1782
K_M190 = 1881


def candidate_scores(legacy, fingerprint, test) -> dict[str, np.ndarray]:
    legacy_equal4 = _equal_component_score(legacy)
    fingerprint_equal4 = _equal_component_score(fingerprint)
    legacy_rank = _rank_score([
        np.mean([legacy[f"{component}_s{seed}"] for seed in BLEND_SEEDS], axis=0)
        for component in BLEND_COMPONENTS
    ])
    gmax = _gmax_score(legacy_equal4, test)
    return {
        "cand_seed42": _equal_component_score(legacy, seeds=[42]),
        "cand_wo_cat_recent": _equal_component_score(legacy, [
            BLEND_COMPONENTS[0], *BLEND_COMPONENTS[2:],
        ]),
        "cand_rank_legacy_fp": _rank_score([legacy_equal4, fingerprint_equal4]),
        "cand_wo_cat_all": _equal_component_score(legacy, BLEND_COMPONENTS[1:]),
        "cand_legacy_rank": legacy_rank,
        "cand_seeds2026_777": _equal_component_score(legacy, seeds=[2026, 777]),
        "cand_rank_all3": _rank_score([legacy_equal4, fingerprint_equal4, gmax]),
    }


def write_candidate(test: pd.DataFrame, champion: np.ndarray, score: np.ndarray,
                    name: str, k: int) -> dict:
    order = np.argsort(-score, kind="stable")
    labels = np.zeros(len(score), dtype=int)
    labels[order[:k]] = 1
    output = pd.DataFrame({"claim_id": test["claim_id"], "is_valid": labels})
    path = ROOT / "submissions" / f"{name}.csv"
    output.to_csv(path, index=False)
    saved = pd.read_csv(path)
    claim_ids = test["claim_id"].astype(str).to_numpy()
    assert len(saved) == 7446
    assert list(saved.columns) == ["claim_id", "is_valid"]
    assert np.array_equal(saved["claim_id"].astype(str).to_numpy(), claim_ids)
    assert set(saved["is_valid"].unique()).issubset({0, 1})
    assert int(saved["is_valid"].sum()) == k
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    overlap = int(np.sum(saved["is_valid"].to_numpy().astype(bool) & champion))
    result = {
        "file": path.name,
        "rows": len(saved),
        "positives": int(saved["is_valid"].sum()),
        "claim_order": True,
        "champion_overlap": overlap,
        "sha256": digest,
    }
    print(result, flush=True)
    return result


def main() -> None:
    legacy = np.load(ROOT / "artifacts" / "probs_test.npz", allow_pickle=True)
    fingerprint = np.load(ROOT / "artifacts" / "probs_test_reproduced.npz", allow_pickle=True)
    test = pd.read_csv(ROOT / "test.csv")
    claim_ids = test["claim_id"].astype(str).to_numpy()
    assert np.array_equal(legacy["claim_id"].astype(str), claim_ids)
    assert np.array_equal(fingerprint["claim_id"].astype(str), claim_ids)

    champion = pd.read_csv(ROOT / "submissions" / "submission_equal4_m180.csv")
    assert np.array_equal(champion["claim_id"].astype(str).to_numpy(), claim_ids)
    champion_top = champion["is_valid"].to_numpy().astype(bool)
    scores = candidate_scores(legacy, fingerprint, test)
    specs = [
        ("cand_seed42", "cand_seed42", K_MAIN),
        ("cand_wo_cat_recent", "cand_wo_cat_recent", K_MAIN),
        ("cand_rank_legacy_fp", "cand_rank_legacy_fp", K_MAIN),
        ("cand_wo_cat_all", "cand_wo_cat_all", K_MAIN),
        ("cand_legacy_rank", "cand_legacy_rank", K_MAIN),
        ("cand_seeds2026_777", "cand_seeds2026_777", K_MAIN),
        ("cand_rank_all3", "cand_rank_all3", K_MAIN),
        ("cand_seed42", "cand_seed42_m190", K_M190),
        ("cand_wo_cat_recent", "cand_wo_cat_recent_m190", K_M190),
        ("cand_rank_legacy_fp", "cand_rank_legacy_fp_m190", K_M190),
    ]
    for score_name, output_name, k in specs:
        write_candidate(test, champion_top, scores[score_name], output_name, k)


if __name__ == "__main__":
    main()
