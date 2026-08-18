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

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
    # `ip_country_id` is dropped by the main pipeline; these two variants measure whether it
    # can be used at all, given that train is 99.4% country 23 while 40% of test is 11.
    "fpte_ip": {"keys": ["fp"], "count": True, "ip_cat": True},
    "fpte_ipte": {"keys": ["fp", "ip"], "count": True},
    "advw": {"advw": True},
    "pseudo025": {},
    "pseudo050": {},
    # Chronological label propagation over the repeated content / owner ids. The base
    # pipeline already has a smoothed K-fold target encoding of these ids; these variants
    # measure the *additional* value of hard, strictly-backward label counters (prior
    # positives, prior negatives, indicators), which K-fold TE smooths away.
    "lbl": {"lbl_keys": ["id_content", "id_content_owner"]},
    "lbl_content": {"lbl_keys": ["id_content"]},
    "lbl_owner": {"lbl_keys": ["id_content_owner"]},
    "lbl_ind": {"lbl_keys": ["id_content", "id_content_owner"], "lbl_ind_only": True},
}
LBL_ALPHA = 5.0
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
        elif name == "ip":
            out[name] = raw["ip_country_id"].astype(str)
        else:
            raise ValueError(name)
    return out


def label_propagation(raw_tr, raw_va, key_col, alpha=LBL_ALPHA, ind_only=False):
    """Strictly-backward label counters over `key_col`.

    Training rows only see labels of *earlier* training rows with the same key (rows are
    already ordered chronologically); validation rows see the whole training prefix.
    """
    y = raw_tr["is_valid"].to_numpy().astype(float)
    keys_tr = raw_tr[key_col].astype(str)
    grouped = pd.DataFrame({"key": keys_tr.to_numpy(), "y": y}).groupby("key", sort=False)
    prior_cnt = grouped.cumcount().to_numpy().astype(float)
    prior_pos = grouped["y"].cumsum().to_numpy() - y
    gm = float(y.mean())
    totals = grouped["y"].agg(["sum", "count"])
    keys_va = raw_va[key_col].astype(str)
    cnt_va = keys_va.map(totals["count"]).fillna(0).to_numpy().astype(float)
    pos_va = keys_va.map(totals["sum"]).fillna(0).to_numpy().astype(float)
    prefix = key_col.replace("id_content_owner", "ownlbl").replace("id_content", "cntlbl")

    def block(cnt, pos):
        neg = cnt - pos
        cols = {
            f"{prefix}_had_pos": (pos > 0).astype(float),
            f"{prefix}_had_neg": (neg > 0).astype(float),
        }
        if not ind_only:
            cols[f"{prefix}_prior_pos"] = pos
            cols[f"{prefix}_prior_neg"] = neg
            cols[f"{prefix}_prior_rate"] = (pos + alpha * gm) / (cnt + alpha)
        return pd.DataFrame(cols)

    return block(prior_cnt, prior_pos), block(cnt_va, pos_va)


def add_variant_features(frame_tr, frame_va, raw_tr, raw_va, hist, variant):
    """Attach the feature block under test to already engineered frames."""
    cfg = VARIANTS[variant]
    extra_numeric: list[str] = []
    if cfg.get("hist") or cfg.get("hist_subset"):
        cols = HIST_SUBSET if cfg.get("hist_subset") else list(hist.columns)
        frame_tr[cols] = hist.loc[raw_tr["claim_id"].to_numpy(), cols].to_numpy()
        frame_va[cols] = hist.loc[raw_va["claim_id"].to_numpy(), cols].to_numpy()
        extra_numeric += cols
    if cfg.get("ip_cat"):
        frame_tr["ip_country_id"] = raw_tr["ip_country_id"].to_numpy()
        frame_va["ip_country_id"] = raw_va["ip_country_id"].to_numpy()
        extra_numeric.append("ip_country_id")
    for key_col in cfg.get("lbl_keys", []):
        blk_tr, blk_va = label_propagation(raw_tr, raw_va, key_col,
                                           ind_only=cfg.get("lbl_ind_only", False))
        for col in blk_tr.columns:
            frame_tr[col] = blk_tr[col].to_numpy()
            frame_va[col] = blk_va[col].to_numpy()
            extra_numeric.append(col)
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


# Model presets. Each LightGBM preset starts from `lgb_neutral` and overrides a few keys;
# `half_life` switches on the exponential recency weights used by the `cat_recent` component.
LGB_PRESETS = {
    "lgb": {},
    "lgb_hl105": {"half_life": 105.0},
    "lgb_hl210": {"half_life": 210.0},
    "lgb_leaves96": {"num_leaves": 96, "min_child_samples": 45},
    "lgb_leaves24": {"num_leaves": 24},
    "lgb_ff60": {"colsample_bytree": 0.60},
    "lgb_lr015": {"learning_rate": 0.015},
    "lgb_mcs80": {"min_child_samples": 80},
}
CAT_PRESETS = {
    "cat": {},
    "cat_hl105": {"half_life": 105.0},
    "cat_depth9": {"depth": 9},
}
PSEUDO_WEIGHTS = {"pseudo025": 0.25, "pseudo050": 0.50}
BLEND_COMPONENTS = ["cat_all", "cat_recent", "lgb_neutral", "lgb_diverse"]
BLEND_SEEDS = [42, 2026, 777]
BLEND_K = 1782
BLEND_GMAX_WC = 0.15
BLEND_GMAX_WO = 0.10
ADV_TIME_COLUMNS = {
    "event_elapsed_days", "hour", "dayofweek", "day", "month", "is_weekend",
    "content_hour", "content_dayofweek", "content_month", "content_age_days",
    "sender_account_age", "claim_user_account_age",
}


def fit_lgb_weighted(x_tr, y_tr, cats, params, seed, iterations, weight=None, eval_set=None):
    """`sol.fit_lightgbm` with optional sample weights (used by the recency presets)."""
    model = lgb.LGBMClassifier(n_estimators=iterations, max_depth=-1, objective="binary",
                               random_state=int(seed), verbosity=-1, n_jobs=sol.THREADS, **params)
    if eval_set is None:
        model.fit(x_tr, y_tr, categorical_feature=cats, sample_weight=weight)
        return model, iterations
    model.fit(x_tr, y_tr, categorical_feature=cats, sample_weight=weight, eval_set=eval_set,
              callbacks=[lgb.early_stopping(sol.ES_ROUNDS, verbose=False)])
    return model, max(int(model.best_iteration_ or iterations), 20)


def adversarial_weights(source, target, seed=42):
    """Estimate mean-one source weights from OOF domain-discriminator probabilities."""
    columns = [
        col for col in source["x_lgb_tr"].columns
        if not col.endswith("_te")
        and not col.endswith("_history_count")
        and col not in ADV_TIME_COLUMNS
    ]
    x_source = source["x_lgb_tr"][columns].copy()
    x_target = target["x_lgb_va"][columns].copy()
    x = pd.concat([x_source, x_target], ignore_index=True)
    cat_cols = [col for col in source["lgb_cats"] if col in columns]
    for col in cat_cols:
        x[col] = x[col].astype("category")
    domain = np.concatenate([np.zeros(len(x_source), dtype=int), np.ones(len(x_target), dtype=int)])
    oof = np.zeros(len(x), dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    for fit_idx, hold_idx in splitter.split(x, domain):
        discriminator = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=24,
            min_child_samples=40,
            objective="binary",
            random_state=int(seed),
            verbosity=-1,
            n_jobs=sol.THREADS,
        )
        discriminator.fit(
            x.iloc[fit_idx],
            domain[fit_idx],
            categorical_feature=cat_cols,
        )
        oof[hold_idx] = discriminator.predict_proba(x.iloc[hold_idx])[:, 1]
    probability = np.clip(oof[:len(x_source)], 1e-6, 1.0 - 1e-6)
    weights = np.clip(probability / (1.0 - probability), 0.05, 20.0)
    return weights / weights.mean()


def _rank_score(arrays):
    """Average descending percentile ranks, preserving stable top-k tie handling."""
    ranks = []
    for values in arrays:
        order = np.argsort(-values, kind="stable")
        rank = np.empty(len(values), dtype=float)
        rank[order] = np.arange(len(values), dtype=float)
        ranks.append(1.0 - rank / max(len(values) - 1, 1))
    return np.mean(ranks, axis=0)


def _equal_component_score(data, components=BLEND_COMPONENTS, seeds=BLEND_SEEDS):
    def component_score(component):
        seeded = [f"{component}_s{seed}" for seed in seeds]
        if all(key in data.files for key in seeded):
            return np.mean([data[key] for key in seeded], axis=0)
        if component in data.files:
            return data[component]
        raise KeyError(f"missing cached probabilities for {component}")

    return np.mean(
        [component_score(component) for component in components],
        axis=0,
    )


def _gmax_score(score, raw_block):
    groups = pd.DataFrame({
        "score": score,
        "content": raw_block["id_content"].astype(str).to_numpy(),
        "owner": raw_block["id_content_owner"].astype(str).to_numpy(),
    })
    content_max = groups.groupby("content")["score"].transform("max").to_numpy()
    owner_max = groups.groupby("owner")["score"].transform("max").to_numpy()
    return (
        (1.0 - BLEND_GMAX_WC - BLEND_GMAX_WO) * score
        + BLEND_GMAX_WC * content_max
        + BLEND_GMAX_WO * owner_max
    )


def run_blend3(train, folds):
    """Evaluate cached probability blends and print CV/test-overlap summaries."""
    test = pd.read_csv(ROOT / "test.csv")
    champion = pd.read_csv(ROOT / "submissions" / "submission_equal4_m180.csv")
    champion_order = champion["claim_id"].astype(str).to_numpy()
    if not np.array_equal(champion_order, test["claim_id"].astype(str).to_numpy()):
        raise ValueError("champion submission order differs from test.csv")
    champion_top = champion["is_valid"].to_numpy().astype(bool)
    legacy_test = np.load(ARTIFACTS / "probs_test.npz", allow_pickle=True)
    fingerprint_test = np.load(ARTIFACTS / "probs_test_reproduced.npz", allow_pickle=True)
    if not np.array_equal(legacy_test["claim_id"].astype(str), champion_order):
        raise ValueError("legacy test probabilities are not in test.csv order")
    if not np.array_equal(fingerprint_test["claim_id"].astype(str), champion_order):
        raise ValueError("fingerprint test probabilities are not in test.csv order")
    test_legacy = _equal_component_score(legacy_test)
    test_fingerprint = _equal_component_score(fingerprint_test)
    test_gmax = _gmax_score(test_legacy, test)
    test_scores = {
        "legacy_prob": test_legacy,
        "legacy_rank": _rank_score([
            np.mean([legacy_test[f"{component}_s{seed}"] for seed in BLEND_SEEDS], axis=0)
            for component in BLEND_COMPONENTS
        ]),
        "legacy_wo_cat_all": _equal_component_score(legacy_test, BLEND_COMPONENTS[1:]),
        "legacy_wo_cat_recent": _equal_component_score(legacy_test, [BLEND_COMPONENTS[0]]
                                                       + BLEND_COMPONENTS[2:]),
        "legacy_wo_lgb_neutral": _equal_component_score(legacy_test, BLEND_COMPONENTS[:2]
                                                        + [BLEND_COMPONENTS[3]]),
        "legacy_wo_lgb_diverse": _equal_component_score(legacy_test, BLEND_COMPONENTS[:3]),
        "legacy_seed42": _equal_component_score(legacy_test, seeds=[42]),
        "legacy_seed2026_777": _equal_component_score(legacy_test, seeds=[2026, 777]),
        "rank_legacy_fingerprint": _rank_score([test_legacy, test_fingerprint]),
        "rank_legacy_gmax": _rank_score([test_legacy, test_gmax]),
        "rank_legacy_fingerprint_gmax": _rank_score([test_legacy, test_fingerprint, test_gmax]),
    }
    rows = []
    raw_sorted = train.reset_index(drop=True)
    for fold in range(folds, 0, -1):
        path = ARTIFACTS / "baseline" / f"probs_fold{fold}.npz"
        fingerprint_path = ARTIFACTS / f"probs_fold{fold}.npz"
        legacy = np.load(path, allow_pickle=True)
        fingerprint = np.load(fingerprint_path, allow_pickle=True)
        raw_start = len(raw_sorted) - fold * BLOCK
        raw_block = raw_sorted.iloc[raw_start:raw_start + BLOCK].reset_index(drop=True)
        if not np.array_equal(legacy["claim_id"].astype(str), raw_block["claim_id"].astype(str).to_numpy()):
            raise ValueError(f"legacy fold {fold} does not match chronological validation block")
        if not np.array_equal(fingerprint["claim_id"].astype(str), legacy["claim_id"].astype(str)):
            raise ValueError(f"fingerprint fold {fold} does not match legacy fold")
        legacy_score = _equal_component_score(legacy)
        fingerprint_score = _equal_component_score(fingerprint)
        gmax_score = _gmax_score(legacy_score, raw_block)
        scores = {
            "legacy_prob": legacy_score,
            "legacy_rank": _rank_score([
                np.mean([legacy[f"{component}_s{seed}"] for seed in BLEND_SEEDS], axis=0)
                for component in BLEND_COMPONENTS
            ]),
            "legacy_wo_cat_all": _equal_component_score(legacy, BLEND_COMPONENTS[1:]),
            "legacy_wo_cat_recent": _equal_component_score(legacy, [BLEND_COMPONENTS[0]]
                                                           + BLEND_COMPONENTS[2:]),
            "legacy_wo_lgb_neutral": _equal_component_score(legacy, BLEND_COMPONENTS[:2]
                                                            + [BLEND_COMPONENTS[3]]),
            "legacy_wo_lgb_diverse": _equal_component_score(legacy, BLEND_COMPONENTS[:3]),
            "legacy_seed42": _equal_component_score(legacy, seeds=[42]),
            "legacy_seed2026_777": _equal_component_score(legacy, seeds=[2026, 777]),
            "rank_legacy_fingerprint": _rank_score([legacy_score, fingerprint_score]),
            "rank_legacy_gmax": _rank_score([legacy_score, gmax_score]),
            "rank_legacy_fingerprint_gmax": _rank_score([legacy_score, fingerprint_score, gmax_score]),
        }
        for variant, score in scores.items():
            metrics = fold_metrics(legacy["y"], score)
            rows.append({
                "model": "blend3",
                "variant": variant,
                "fold": fold,
                "iterations": 0,
                "seeds": "cached",
                "seconds": 0.0,
                **metrics,
            })
    result = pd.DataFrame(rows)
    summary = result.groupby("variant")[["roc_auc", "pr_auc", "f1_m1.8"]].mean()
    for variant, score in test_scores.items():
        top = np.zeros(len(score), dtype=bool)
        top[np.argsort(-score, kind="stable")[:BLEND_K]] = True
        overlap = int(np.sum(top & champion_top))
        print(json.dumps({"variant": variant, "test_top_k_overlap": overlap,
                          "test_top_k_overlap_fraction": round(overlap / BLEND_K, 6)}), flush=True)
    print(result.sort_values(["variant", "fold"]).to_string(index=False), flush=True)
    print(summary.round(6).to_string(), flush=True)
    return result


def score_fold(raw_tr, raw_va, mappings, hist, variant, model, seeds):
    """Train on the prefix, score the validation block; iterations from an inner holdout."""
    feature_variant = "base" if variant == "advw" or variant in PSEUDO_WEIGHTS else variant
    inner_cut = max(len(raw_tr) - BLOCK, int(0.6 * len(raw_tr)))
    inner = build(raw_tr.iloc[:inner_cut], raw_tr.iloc[inner_cut:], mappings, hist, feature_variant)
    outer = build(raw_tr, raw_va, mappings, hist, feature_variant)
    inner_times = pd.to_datetime(raw_tr.iloc[:inner_cut]["first_event_time"])
    outer_times = pd.to_datetime(raw_tr["first_event_time"])
    is_pseudo = variant in PSEUDO_WEIGHTS
    if model in LGB_PRESETS:
        overrides = dict(LGB_PRESETS[model])
        half_life = overrides.pop("half_life", None)
        params = {**sol.LGB_PARAMS["lgb_neutral"], **overrides}
        w_inner = None if half_life is None else sol.recency_weight(inner_times, half_life)
        w_outer = None if half_life is None else sol.recency_weight(outer_times, half_life)
        if variant == "advw":
            w_inner_adv = adversarial_weights(inner, inner, seed=seeds[0])
            w_outer_adv = adversarial_weights(outer, outer, seed=seeds[0])
            w_inner = w_inner_adv if w_inner is None else w_inner * w_inner_adv
            w_outer = w_outer_adv if w_outer is None else w_outer * w_outer_adv
        _, iterations = fit_lgb_weighted(
            inner["x_lgb_tr"], inner["y_tr"], inner["lgb_cats"], params, seeds[0],
            sol.MAX_ITERATIONS_LGB, weight=w_inner,
            eval_set=[(inner["x_lgb_va"], inner["y_va"])])
        initial = {}
        for seed in seeds:
            fitted, _ = fit_lgb_weighted(outer["x_lgb_tr"], outer["y_tr"], outer["lgb_cats"],
                                         params, seed, iterations, weight=w_outer)
            initial[seed] = fitted.predict_proba(outer["x_lgb_va"])[:, 1]
        if is_pseudo:
            ranking = np.mean(list(initial.values()), axis=0)
            target_order = np.argsort(-ranking, kind="stable")
            n_positive = max(1, min(len(target_order) - 1,
                                    int(round(sol.RATE_MULTIPLIER * outer["y_tr"].mean()
                                              * len(target_order)))))
            n_negative = max(1, len(target_order) // 2)
            pseudo_positive = target_order[:n_positive]
            pseudo_negative = target_order[-n_negative:]
            pseudo_idx = np.concatenate([pseudo_positive, pseudo_negative])
            pseudo_y = np.concatenate([
                np.ones(len(pseudo_positive), dtype=int),
                np.zeros(len(pseudo_negative), dtype=int),
            ])
            x_tr = pd.concat([
                outer["x_lgb_tr"],
                outer["x_lgb_va"].iloc[pseudo_idx],
            ], ignore_index=True)
            for col in outer["lgb_cats"]:
                categories = outer["x_lgb_tr"][col].cat.categories
                x_tr[col] = x_tr[col].astype(
                    pd.api.types.CategoricalDtype(categories=categories)
                )
            y_tr = np.concatenate([outer["y_tr"], pseudo_y])
            base_weight = np.ones(len(outer["y_tr"])) if w_outer is None else w_outer
            weight = np.concatenate([
                base_weight,
                np.full(len(pseudo_idx), PSEUDO_WEIGHTS[variant]),
            ])
            probs = {}
            for seed in seeds:
                fitted, _ = fit_lgb_weighted(
                    x_tr, y_tr, outer["lgb_cats"], params, seed, iterations, weight=weight)
                probs[seed] = fitted.predict_proba(outer["x_lgb_va"])[:, 1]
        else:
            probs = initial
    elif model in CAT_PRESETS:
        cfg = {**sol.CAT_PARAMS["cat_all"], **CAT_PRESETS[model]}
        w_inner = None if cfg["half_life"] is None else sol.recency_weight(inner_times, cfg["half_life"])
        w_outer = None if cfg["half_life"] is None else sol.recency_weight(outer_times, cfg["half_life"])
        if variant == "advw":
            w_inner_adv = adversarial_weights(inner, inner, seed=seeds[0])
            w_outer_adv = adversarial_weights(outer, outer, seed=seeds[0])
            w_inner = w_inner_adv if w_inner is None else w_inner * w_inner_adv
            w_outer = w_outer_adv if w_outer is None else w_outer * w_outer_adv
        _, iterations = sol.fit_catboost(
            inner["x_cat_tr"], inner["y_tr"], inner["cats"], w_inner, cfg["l2_leaf_reg"],
            cfg["depth"], seeds[0], sol.MAX_ITERATIONS_CAT,
            eval_set=(inner["x_cat_va"], inner["y_va"]))
        initial = {}
        for seed in seeds:
            fitted, _ = sol.fit_catboost(outer["x_cat_tr"], outer["y_tr"], outer["cats"], w_outer,
                                         cfg["l2_leaf_reg"], cfg["depth"], seed, iterations)
            initial[seed] = fitted.predict_proba(outer["x_cat_va"])[:, 1]
        if is_pseudo:
            ranking = np.mean(list(initial.values()), axis=0)
            target_order = np.argsort(-ranking, kind="stable")
            n_positive = max(1, min(len(target_order) - 1,
                                    int(round(sol.RATE_MULTIPLIER * outer["y_tr"].mean()
                                              * len(target_order)))))
            n_negative = max(1, len(target_order) // 2)
            pseudo_positive = target_order[:n_positive]
            pseudo_negative = target_order[-n_negative:]
            pseudo_idx = np.concatenate([pseudo_positive, pseudo_negative])
            pseudo_y = np.concatenate([
                np.ones(len(pseudo_positive), dtype=int),
                np.zeros(len(pseudo_negative), dtype=int),
            ])
            x_tr = pd.concat([
                outer["x_cat_tr"],
                outer["x_cat_va"].iloc[pseudo_idx],
            ], ignore_index=True)
            y_tr = np.concatenate([outer["y_tr"], pseudo_y])
            base_weight = np.ones(len(outer["y_tr"])) if w_outer is None else w_outer
            weight = np.concatenate([
                base_weight,
                np.full(len(pseudo_idx), PSEUDO_WEIGHTS[variant]),
            ])
            probs = {}
            for seed in seeds:
                fitted, _ = sol.fit_catboost(
                    x_tr, y_tr, outer["cats"], weight, cfg["l2_leaf_reg"], cfg["depth"], seed, iterations)
                probs[seed] = fitted.predict_proba(outer["x_cat_va"])[:, 1]
        else:
            probs = initial
    mean_prob = np.mean(list(probs.values()), axis=0)
    return outer["y_va"], mean_prob, iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="base")
    parser.add_argument("--model", default="lgb")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--blend3", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ARTIFACTS.mkdir(exist_ok=True)

    train = pd.read_csv(ROOT / "train.csv").sort_values("first_event_time").reset_index(drop=True)
    test = pd.read_csv(ROOT / "test.csv")
    if args.blend3:
        result = run_blend3(train, args.folds)
        path = ARTIFACTS / "exp_features.csv"
        if path.exists():
            result = pd.concat([pd.read_csv(path), result], ignore_index=True)
        result.to_csv(path, index=False)
        return
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
