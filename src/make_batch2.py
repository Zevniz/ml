"""Generate the second cached consensus-selection batch.

No models are trained. All scores come from the saved legacy and fingerprint
test probability caches:

    python src/make_batch2.py
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

K = 1782
GMAX_WEIGHTS = {
    "standard": (0.15, 0.10),
    "hi": (0.25, 0.15),
    "lo": (0.08, 0.05),
}


def legacy_seed_score(data, seeds) -> np.ndarray:
    return _equal_component_score(data, seeds=seeds)


def rank_all3(legacy, fingerprint, test, wc=0.15, wo=0.10) -> np.ndarray:
    legacy_score = _equal_component_score(legacy, seeds=BLEND_SEEDS)
    fingerprint_score = _equal_component_score(fingerprint, seeds=BLEND_SEEDS)
    gmax_score = _gmax_score(legacy_score, test, wc=wc, wo=wo)
    return _rank_score([legacy_score, fingerprint_score, gmax_score])


def vote_top4(legacy, fingerprint, test) -> np.ndarray:
    scores = [
        _equal_component_score(legacy, seeds=BLEND_SEEDS),
        _equal_component_score(fingerprint, seeds=BLEND_SEEDS),
        _gmax_score(_equal_component_score(legacy, seeds=BLEND_SEEDS), test),
        legacy_seed_score(legacy, [2026, 777]),
    ]
    indicators = np.asarray([
        np.isin(np.arange(len(score)), np.argsort(-score, kind="stable")[:K])
        for score in scores
    ])
    votes = indicators.sum(axis=0)
    rank_tie_break = _rank_score(scores)
    order = np.lexsort((np.arange(len(votes)), -rank_tie_break, -votes))
    result = np.zeros(len(votes), dtype=float)
    result[order[:K]] = 1.0
    return result


def candidate_scores(legacy, fingerprint, test) -> dict[str, np.ndarray]:
    legacy_score = _equal_component_score(legacy, seeds=BLEND_SEEDS)
    fingerprint_score = _equal_component_score(fingerprint, seeds=BLEND_SEEDS)
    gmax_score = _gmax_score(legacy_score, test)
    seed42_score = legacy_seed_score(legacy, [42])
    seeds_subset_score = legacy_seed_score(legacy, [2026, 777])
    legacy_equal5 = _equal_component_score(
        legacy, BLEND_COMPONENTS + ["lgb_strong"], seeds=BLEND_SEEDS
    )
    return {
        "cand_rank_top5": _rank_score([
            legacy_score, fingerprint_score, gmax_score,
            seed42_score, seeds_subset_score,
        ]),
        "cand_vote_top4": vote_top4(legacy, fingerprint, test),
        "cand_gmax_hi": rank_all3(legacy, fingerprint, test, *GMAX_WEIGHTS["hi"]),
        "cand_gmax_lo": rank_all3(legacy, fingerprint, test, *GMAX_WEIGHTS["lo"]),
        "cand_rank_wide": _rank_score([
            legacy_score, fingerprint_score, gmax_score,
            seed42_score, seeds_subset_score, legacy_equal5,
        ]),
    }


def write_candidate(test: pd.DataFrame, champion: np.ndarray, rank_all3_top: np.ndarray,
                    score: np.ndarray, name: str, existing_hashes: dict[str, str]) -> dict:
    order = np.argsort(-score, kind="stable")
    labels = np.zeros(len(score), dtype=int)
    labels[order[:K]] = 1
    output = pd.DataFrame({"claim_id": test["claim_id"], "is_valid": labels})
    path = ROOT / "submissions" / f"{name}.csv"
    output.to_csv(path, index=False)
    saved = pd.read_csv(path)
    claim_ids = test["claim_id"].astype(str).to_numpy()
    assert len(saved) == 7446
    assert list(saved.columns) == ["claim_id", "is_valid"]
    assert np.array_equal(saved["claim_id"].astype(str).to_numpy(), claim_ids)
    assert set(saved["is_valid"].unique()).issubset({0, 1})
    assert int(saved["is_valid"].sum()) == K
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    duplicate_of = [filename for filename, old_digest in existing_hashes.items() if old_digest == digest]
    assert not duplicate_of, f"{path.name} duplicates existing submission(s): {duplicate_of}"
    existing_hashes[path.name] = digest
    selected = saved["is_valid"].to_numpy().astype(bool)
    result = {
        "file": path.name,
        "rows": len(saved),
        "positives": int(selected.sum()),
        "claim_order": True,
        "champion_overlap": int(np.sum(selected & champion)),
        "rank_all3_overlap": int(np.sum(selected & rank_all3_top)),
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
    rank_all3 = pd.read_csv(ROOT / "submissions" / "cand_rank_all3.csv")
    assert np.array_equal(champion["claim_id"].astype(str).to_numpy(), claim_ids)
    assert np.array_equal(rank_all3["claim_id"].astype(str).to_numpy(), claim_ids)
    champion_top = champion["is_valid"].to_numpy().astype(bool)
    rank_all3_top = rank_all3["is_valid"].to_numpy().astype(bool)
    existing_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "submissions").glob("*.csv")
        if not path.name.startswith("cand_rank_top5")
        and not path.name.startswith("cand_vote_top4")
        and not path.name.startswith("cand_gmax_hi")
        and not path.name.startswith("cand_gmax_lo")
        and not path.name.startswith("cand_rank_wide")
    }
    scores = candidate_scores(legacy, fingerprint, test)
    for name, score in scores.items():
        write_candidate(test, champion_top, rank_all3_top, score, name, existing_hashes)


if __name__ == "__main__":
    main()
