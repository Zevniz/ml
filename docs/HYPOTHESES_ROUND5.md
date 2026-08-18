# Round 5 — propagation over repeated `id_content` / `id_content_owner`

Motivation: this was the last unexplored large structure in the data. On train, 3,169
`id_content` groups (9,909 rows) and 6,742 `id_content_owner` groups (34,900 rows) repeat,
83.6% / 70.1% of the repeated groups are pure by target, and on test 1,382 rows share an
`id_content` with train and 3,706 rows share an `id_content_owner`.

Setup identical to the earlier rounds: 4 chronological folds of 7,446 rows, expanding
training prefix, inner holdout for iteration selection, seed 42, metric = mean top-k F1 at
m=1.8 across the folds.

## 1. Hard chronological label propagation — rejected

Strictly-backward label counters per key (prior positives, prior negatives, smoothed prior
rate `alpha=5`, had-positive / had-negative indicators), where each training row only sees
labels of earlier training rows and validation rows see the whole prefix
(`src/exp_features.py`, variants `lbl`, `lbl_content`, `lbl_owner`, `lbl_ind`).

| model | variant | mean F1 m=1.8 | vs base |
|---|---|---:|---:|
| lgb | base | 0.4152 | — |
| lgb | lbl | 0.4148 | −0.0004 |
| lgb | lbl_content | 0.4151 | −0.0001 |
| lgb | lbl_owner | 0.4154 | +0.0002 |
| lgb | lbl_ind | 0.4122 | −0.0030 |
| cat | base | 0.4125 | — |
| cat | lbl | 0.4111 | −0.0014 |
| cat | lbl_content | 0.4131 | +0.0006 |
| cat | lbl_ind | 0.4138 | +0.0013 (PR AUC worse) |
| cat | lbl_owner | 0.4103 | −0.0022 |

Conclusion: the shipped K-fold target encodings of `id_content` / `id_content_owner`
already absorb this signal; the hard counters add nothing beyond noise. Rejected.

## 2. Within-group probability *mean* smoothing — rejected

Replacing each row's blend probability with a mix towards the mean of its `id_content`
(or owner) group inside the scored block: monotonically worse the stronger the mix
(0.4173 → 0.4113 at 50% mix on the lgb+cat blend). Groups mix valid and invalid claims;
averaging drags positives down. Rejected.

## 3. Within-group probability *max* propagation — accepted as leaderboard probe

Mixing each row's equal4-blend probability towards the *maximum* of its group, computed
over the scored block only (no labels ⇒ no leakage; at inference the groups are formed
over the 7,446 test rows):

```
p_adj = (1 - wc - wo) * p + wc * max over id_content + wo * max over id_content_owner
```

Grid over `wc`, `wo` on the full equal4 ensemble OOF probabilities
(`artifacts/probs_fold*.npz`):

| wc | wo | mean F1 m=1.8 | per fold |
|---:|---:|---:|---|
| 0 | 0 | 0.4206 | 0.4150 / 0.4167 / 0.4179 / 0.4327 |
| 0.10 | 0.10 | 0.4230 | 0.4179 / 0.4167 / 0.4234 / 0.4341 |
| 0.15 | 0.10 | 0.4232 | 0.4179 / 0.4167 / 0.4234 / 0.4348 |
| 0.20 | 0.10 | 0.4234 | 0.4186 / 0.4167 / 0.4234 / 0.4348 |

The chosen interior point `wc=0.15, wo=0.10` is never worse than the base on any fold and
also holds at m=1.6 / m=2.0. Probe file: `submissions/submission_equal4_m180_gmax.csv`
(`python src/make_gmax_probe.py`, k=1782, 1,748 of 1,782 positives shared with the
champion, 34 rows swapped). Expected effect from CV: ≈ +0.0026 F1; maximum possible
effect from 34 swaps: ±0.0245.

Caveat: the reporter fingerprint also gained ≈ +0.003 on this CV and lost on the
leaderboard, so this must be confirmed online before it is shipped. The shipped
`submission.csv` remains the k=1782 champion until then.

### Leaderboard result — rejected

`submission_equal4_m180_gmax.csv` scored **0.488** online vs the champion's 0.489 at the
same k=1782 (≈ 676 vs 678 true positives): the offline +0.0026 does not transfer, exactly
like the reporter fingerprint before it. The 34 swapped rows net-lose about 2 true
positives. Group-max propagation is rejected; the shipped `submission.csv` stays the
k=1782 champion at 0.489.
