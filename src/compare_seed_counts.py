"""Does averaging more seeds help? Measured on identical folds, one training run.

Every component is stored per seed (`{name}_s{seed}`), so the k-seed arm is just the
mean over the first k seeds of the same cached run. That makes the comparison exactly
controlled: same folds, same feature matrices, same early-stopping iteration counts
(the inner holdout fit always uses `seeds[0]`), same library versions. The only thing
that varies is how many seeds enter the average.

Usage:
    python src/compare_seed_counts.py --dir /tmp/vk_seed12 \
        --seeds 42,2026,777,1337,7,13,99,123,2024,31337,555,8888 --counts 1,3,6,12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ["cat_all", "cat_recent", "lgb_neutral", "lgb_diverse"]
MULTIPLIERS = np.round(np.arange(1.4, 2.25, 0.1), 2)
REFERENCE_MULTIPLIER = 1.8  # the shipped operating point


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    labels = np.zeros(len(scores), dtype=int)
    labels[np.argsort(-scores, kind="stable")[:k]] = 1
    return labels


def equal4_blend(data, seeds: list[int]) -> np.ndarray:
    """Equal-weight blend of the four shipped components, averaged over `seeds`."""
    return np.mean([np.mean([data[f"{c}_s{s}"] for s in seeds], axis=0)
                    for c in COMPONENTS], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="directory holding probs_fold*.npz")
    parser.add_argument("--seeds", type=str, required=True, help="seed list in run order")
    parser.add_argument("--counts", type=str, default="1,3,6,12", help="seed-count arms to compare")
    parser.add_argument("--folds", type=str, default="1,2,3,4")
    args = parser.parse_args()

    all_seeds = [int(s) for s in args.seeds.split(",")]
    counts = [int(c) for c in args.counts.split(",")]
    folds = [int(f) for f in args.folds.split(",")]
    source = Path(args.dir)

    rows: list[dict] = []
    for fold in folds:
        data = np.load(source / f"probs_fold{fold}.npz", allow_pickle=True)
        y = data["y"].astype(int)
        size = len(y)
        prevalence = float(y.mean())
        for count in counts:
            seeds = all_seeds[:count]
            scores = equal4_blend(data, seeds)
            record = {
                "fold": fold,
                "seeds": count,
                "prevalence": round(prevalence, 4),
                "roc_auc": round(float(roc_auc_score(y, scores)), 5),
                "pr_auc": round(float(average_precision_score(y, scores)), 5),
            }
            for multiplier in MULTIPLIERS:
                k = int(round(prevalence * multiplier * size))
                record[f"m{multiplier:.1f}"] = round(float(f1_score(y, top_k(scores, k))), 5)
            rows.append(record)

    table = pd.DataFrame(rows)
    out = ROOT / "artifacts" / "analysis_seed_counts.csv"
    table.to_csv(out, index=False)

    reference = f"m{REFERENCE_MULTIPLIER:.1f}"
    multiplier_columns = [f"m{m:.1f}" for m in MULTIPLIERS]
    pd.set_option("display.width", 250)

    print("== per fold, F1 at the shipped operating point m=1.8 ==")
    pivot = table.pivot(index="fold", columns="seeds", values=reference)
    print(pivot.to_string())

    print("\n== mean over folds ==")
    summary = table.groupby("seeds")[["roc_auc", "pr_auc"] + multiplier_columns].mean().round(5)
    print(summary[["roc_auc", "pr_auc", reference]].to_string())

    baseline = counts[0] if len(counts) == 1 else 3 if 3 in counts else counts[0]
    print(f"\n== delta versus the {baseline}-seed arm (the shipped configuration) ==")
    base = summary.loc[baseline]
    for count in counts:
        if count == baseline:
            continue
        row = summary.loc[count]
        print(f"  {count:2d} seeds: "
              f"pr_auc {row['pr_auc'] - base['pr_auc']:+.5f}   "
              f"roc_auc {row['roc_auc'] - base['roc_auc']:+.5f}   "
              f"F1@m1.8 {row[reference] - base[reference]:+.5f}")

    print(f"\n== F1@m1.8 per fold, delta versus {baseline} seeds "
          f"(sign consistency matters more than size) ==")
    for count in counts:
        if count == baseline:
            continue
        deltas = [pivot.loc[f, count] - pivot.loc[f, baseline] for f in folds]
        wins = sum(1 for d in deltas if d > 0)
        print(f"  {count:2d} seeds: " + "  ".join(f"fold{f} {d:+.5f}" for f, d in zip(folds, deltas))
              + f"   | improved on {wins}/{len(folds)} folds")

    print(f"\n== full multiplier curve, mean over folds ==")
    print(summary[multiplier_columns].to_string())
    print(f"\nwritten: {out.relative_to(ROOT)}")
    print("Repository acceptance gate for a hypothesis: dF1 >= +0.0025 and dAUC >= 0.")


if __name__ == "__main__":
    main()
