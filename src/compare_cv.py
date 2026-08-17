"""Compare two cached CV runs on the equal-weight 4-component blend.

    python src/compare_cv.py artifacts/baseline artifacts

Both directories must contain `probs_fold{1..4}.npz` produced by `src/cv_probs.py` with the
same folds and seeds. Only the decision rule of `solution.py` (top-k with k = m * expected
positives) is applied here; nothing is fitted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_utils import f1_at_m  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

COMPONENTS = ["cat_all", "cat_recent", "lgb_neutral", "lgb_diverse"]


def blend(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    y = data["y"]
    parts = []
    for component in COMPONENTS:
        keys = [k for k in data.files if k.startswith(component + "_s")]
        if not keys:
            raise KeyError(f"{component} missing in {path}")
        parts.append(np.mean([data[k] for k in keys], axis=0))
    return y, np.mean(parts, axis=0)


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or [ROOT / "artifacts"]
    print(f"{'fold':>5} " + " ".join(f"{d.name:>28}" for d in dirs))
    totals = {d: [] for d in dirs}
    for fold in range(4, 0, -1):
        cells = []
        for directory in dirs:
            y, score = blend(directory / f"probs_fold{fold}.npz")
            f1 = f1_at_m(y, score, 1.8)
            totals[directory].append((f1, roc_auc_score(y, score), average_precision_score(y, score)))
            cells.append(f"f1={f1:.4f} auc={roc_auc_score(y, score):.4f} pr={average_precision_score(y, score):.4f}")
        print(f"{fold:>5} " + " ".join(f"{c:>28}" for c in cells))
    print("\nmeans")
    for directory in dirs:
        arr = np.array(totals[directory])
        print(f"  {directory}: f1_m1.8={arr[:, 0].mean():.4f} roc_auc={arr[:, 1].mean():.4f} pr_auc={arr[:, 2].mean():.4f}")


if __name__ == "__main__":
    main()
