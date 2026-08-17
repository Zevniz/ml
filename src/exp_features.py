"""Paired feature ablation on the 4 chronological folds.

Every variant reuses exactly the fold definition, the inner-holdout iteration selection and
the model hyper-parameters of `src/cv_probs.py`, so the only difference between two runs is
the feature block under test. Results are appended to `artifacts/exp_features.csv`.

    python src/exp_features.py --variant base --model lgb --seeds 42
    python src/exp_features.py --variant hist,fpte,both --model lgb,cat --seeds 42,2026
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import features_v2 as fv2  # noqa: E402
import solution as sol  # noqa: E402
from eval_utils import fold_metrics  # noqa: E402

BLOCK = sol.BLOCK
ARTIFACTS = ROOT / "artifacts"
HIST_CACHE = ARTIFACTS / "history_features.parquet"


def history_features(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    if HIST_CACHE.exists():
        return pd.read_parquet(HIST_CACHE)
    frame = fv2.transductive_history(train, test)
    frame.to_parquet(HIST_CACHE)
    return frame


# Feature blocks under test. `fp_extra` adds columns to the fingerprint tuple, `keys` lists
# the fingerprint-derived keys that get an out-of-fold target encoding, `count` keeps the
# prefix frequency of the key, `hist` attaches the backward-looking history block.
VARIANTS = {
    "base": {},
    "hist": {"hist": True},
    "fpte": {"keys": ["fp"], "count": True},
    "both": {"hist": True, "keys": ["fp"], "count": True},
    "fpte_nocount": {"keys": ["fp"]},
    "fpte_a40": {"keys": ["fp"], "count": True, "alpha": 40.0},
    "fpte_a5": {"keys": ["fp"], "count": True, "alpha": 5.0},
    "fp11te": {"keys": ["fp"], "count": True, "fp_extra": ["friends_bucket"]},
    "fpte_type": {"keys": ["fp", "fp_type"], "count": True},
    "fpte_reason": {"keys": ["fp", "fp_reason"], "count": True},
    "fpte_hist": {"keys": ["fp"], "count": True, "hist_subset": True},
}
HIST_SUBSET = ["fp_prior_claims", "fp_gap_prev_hours", "fp_owner_prior_claims",
               "content_prior_claims", "owner_prior_claims"]


def _fp_keys(raw: pd.DataFrame, extra: list[str], names: list[str]) -> dict[str, pd.Series]:
    cols = fv2.FP_COLS + extra
    fp = raw[cols].astype(str).agg("|".join, axis=1)
    out = {}
    for name in names:
        if name == "fp":
            out[name] = fp
        elif name == "fp_type":
            out[name] = fp + "||" + raw["claim_type"].astype(str)
        elif name == "fp_reason":
            out[name] = fp + "||" + raw["claim_reason_start"].astype(str)
        else:
            raise ValueError(name)
    return out


def add_variant_features(frame_tr, frame_va, raw_tr, raw_va, hist, variant):
    """Attach the feature block under test to already engineered frames."""
    cfg = VARIANTS[variant]
    extra_numeric: list[str] = []
    if cfg.get("hist") or cfg.get("hist_subset"):
        cols = HIST_SUBSET if cfg.get("hist_subset") else list(hist.columns)
        frame_tr[cols] = hist.loc[raw_tr["claim_id"].to_numpy(), cols].to_numpy()
        frame_va[cols] = hist.loc[raw_va["claim_id"].to_numpy(), cols].to_numpy()
        extra_numeric += cols
    keys = cfg.get("keys", [])
    if keys:
        y_tr = raw_tr["is_valid"].to_numpy()
        keys_tr = _fp_keys(raw_tr, cfg.get("fp_extra", []), keys)
        keys_va = _fp_keys(raw_va, cfg.get("fp_extra", []), keys)
        for name in keys:
            te_tr, te_va, cnt_tr, cnt_va = fv2.oof_target_encoding(
                keys_tr[name], y_tr, keys_va[name], alpha=cfg.get("alpha", fv2.TE_ALPHA))
            frame_tr[f"{name}_te"], frame_va[f"{name}_te"] = te_tr, te_va
            extra_numeric.append(f"{name}_te")
            if cfg.get("count"):
                frame_tr[f"{name}_history_count"] = cnt_tr
                frame_va[f"{name}_history_count"] = cnt_va
                extra_numeric.append(f"{name}_history_count")
    return frame_tr, frame_va, extra_numeric


def build(raw_tr, raw_va, mappings, hist, variant):
    frame_tr, frame_va, no_id, cat_features, cats = sol.prepare_features(raw_tr, raw_va, mappings)
    frame_tr, frame_va, extra = add_variant_features(frame_tr, frame_va, raw_tr, raw_va, hist, variant)
    no_id = no_id + extra
    cat_features = cat_features + extra
    x_cat_tr = sol.prepare_cat(frame_tr, cat_features, cats)
    x_cat_va = sol.prepare_cat(frame_va, cat_features, cats)
    lgb_cats = [c for c in sol.CAT_FEATURES if c in no_id]
    x_lgb_tr, x_lgb_va, lgb_cat_cols = sol.encode_lgb(frame_tr, frame_va, no_id, lgb_cats)
    return {
        "x_cat_tr": x_cat_tr, "x_cat_va": x_cat_va, "cats": cats,
        "x_lgb_tr": x_lgb_tr, "x_lgb_va": x_lgb_va, "lgb_cats": lgb_cat_cols,
        "y_tr": frame_tr["is_valid"].to_numpy(),
        "y_va": frame_va["is_valid"].to_numpy() if "is_valid" in frame_va else None,
    }


def score_fold(raw_tr, raw_va, mappings, hist, variant, model, seeds):
    """Train on the prefix, score the validation block; iterations from an inner holdout."""
    inner_cut = max(len(raw_tr) - BLOCK, int(0.6 * len(raw_tr)))
    inner = build(raw_tr.iloc[:inner_cut], raw_tr.iloc[inner_cut:], mappings, hist, variant)
    outer = build(raw_tr, raw_va, mappings, hist, variant)
    probs = {}
    if model == "lgb":
        params = sol.LGB_PARAMS["lgb_neutral"]
        _, iterations = sol.fit_lightgbm(
            inner["x_lgb_tr"], inner["y_tr"], inner["lgb_cats"], params, seeds[0],
            sol.MAX_ITERATIONS_LGB, eval_set=[(inner["x_lgb_va"], inner["y_va"])])
        for seed in seeds:
            fitted, _ = sol.fit_lightgbm(outer["x_lgb_tr"], outer["y_tr"], outer["lgb_cats"],
                                         params, seed, iterations)
            probs[seed] = fitted.predict_proba(outer["x_lgb_va"])[:, 1]
    else:
        cfg = sol.CAT_PARAMS["cat_all"]
        inner_times = pd.to_datetime(raw_tr.iloc[:inner_cut]["first_event_time"])
        outer_times = pd.to_datetime(raw_tr["first_event_time"])
        w_inner = None if cfg["half_life"] is None else sol.recency_weight(inner_times, cfg["half_life"])
        w_outer = None if cfg["half_life"] is None else sol.recency_weight(outer_times, cfg["half_life"])
        _, iterations = sol.fit_catboost(
            inner["x_cat_tr"], inner["y_tr"], inner["cats"], w_inner, cfg["l2_leaf_reg"],
            cfg["depth"], seeds[0], sol.MAX_ITERATIONS_CAT,
            eval_set=(inner["x_cat_va"], inner["y_va"]))
        for seed in seeds:
            fitted, _ = sol.fit_catboost(outer["x_cat_tr"], outer["y_tr"], outer["cats"], w_outer,
                                         cfg["l2_leaf_reg"], cfg["depth"], seed, iterations)
            probs[seed] = fitted.predict_proba(outer["x_cat_va"])[:, 1]
    mean_prob = np.mean(list(probs.values()), axis=0)
    return outer["y_va"], mean_prob, iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="base")
    parser.add_argument("--model", default="lgb")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ARTIFACTS.mkdir(exist_ok=True)

    train = pd.read_csv(ROOT / "train.csv").sort_values("first_event_time").reset_index(drop=True)
    test = pd.read_csv(ROOT / "test.csv")
    mappings = sol.compute_agg_counts(train, test)
    hist = history_features(train, test)

    rows = []
    total = len(train)
    for model in args.model.split(","):
        for variant in args.variant.split(","):
            for fold in range(args.folds, 0, -1):
                start = total - fold * BLOCK
                raw_tr = train.iloc[:start].copy()
                raw_va = train.iloc[start:start + BLOCK].copy()
                started = time.time()
                y, prob, iterations = score_fold(raw_tr, raw_va, mappings, hist, variant, model, seeds)
                metrics = fold_metrics(y, prob)
                record = {"model": model, "variant": variant, "fold": fold, "iterations": iterations,
                          "seeds": ",".join(map(str, seeds)), "seconds": round(time.time() - started, 1),
                          **metrics}
                rows.append(record)
                np.save(ARTIFACTS / f"exp_{model}_{variant}_fold{fold}.npy", prob)
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                  for k, v in record.items()}), flush=True)
    frame = pd.DataFrame(rows)
    path = ARTIFACTS / "exp_features.csv"
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)
    summary = frame.groupby(["model", "variant"])[["roc_auc", "pr_auc", "f1_m1.8"]].mean().round(4)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
