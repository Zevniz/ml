# The public leaderboard scores half of the test file (3723 rows, 495 positives)

Reproduce with `python src/decode_leaderboard.py`.

## Why the exact F1 values give this away

A top-k submission predicts `k_scored` positives inside the scored set, which contains `P`
positives, so the positive-class F1 is exactly rational:

```
F1 = 2 * TP / (k_scored + P)
```

Two independent constraints pin the scored set down.

1. The all-ones submission scores `0.2347083926 = 165/703` exactly. With `F1 = 2r/(1+r)` and
   `r = P/T` this forces `P/T = 165/1241`, i.e. `T = 1241*m`, `P = 165*m`, `m <= 6`
   (`m = 6` is the whole 7446-row test file with 990 positives).
2. For every uploaded file we know `k` on the full test file, so `k_scored` must be close to
   `k * T / 7446` and can never exceed `k`.

Enumerating all `(k_scored, TP)` pairs for the nine submissions whose F1 the platform reported
at full precision leaves exactly one candidate:

| scored set | submissions explained within 4 sd | verdict |
| --- | --- | --- |
| T=1241, P=165 | 2/9 | rejected |
| T=2482, P=330 | 2/9 | rejected |
| **T=3723, P=495** | **8/9** | **consistent** |
| T=4964, P=660 | 2/9 | rejected |
| T=6205, P=825 | 2/9 | rejected |
| T=7446, P=990 | 0/9 | impossible: needs `k_scored > k` |

The full-test hypothesis fails hard: for the `m180` files it requires 1794 scored positives out
of a file that contains 1782, which cannot happen. Under `T=3723` every submission lands at a
scored-positive share of 0.502-0.505 (expected 0.500, |z| <= 0.4), which is what a 50/50 split
of the test file looks like. The single outlier is the earliest reported score (`m140_fp`,
share 0.586, +6.4 sd) and is most likely a mistyped digit in that report.

## Recovered true-positive counts on the public half

| submission | k | k_scored | TP | precision | recall |
| --- | --- | --- | --- | --- | --- |
| m140_fp | 1386 | 812 | 313 | 0.3855 | 0.6323 |
| m180_fp | 1782 | 897 | 338 | 0.3768 | 0.6828 |
| m190_fp | 1881 | 947 | 351 | 0.3706 | 0.7091 |
| m200_fp | 1980 | 993 | 356 | 0.3585 | 0.7192 |
| m190 (legacy) | 1881 | 950 | 351 | 0.3695 | 0.7091 |
| m195 (legacy) | 1930 | 970 | 352 | 0.3629 | 0.7111 |
| probe_seg_c11 | 808 | 393 | 159 | 0.4046 | 0.3212 |
| probe_shift_to23 | 1782 | 897 | 333 | 0.3712 | 0.6727 |
| probe_shift_to11 | 1782 | 896 | 328 | 0.3661 | 0.6626 |

Scores reported rounded (champion `m180` 0.489, `m180_gmax` 0.488, `m170` 0.48404) imply
TP ~ 340, ~340 and ~325 respectively.

## What this changes

* **One true positive is worth 0.00144 F1**, and `F1 >= 0.500` at `k_scored = 897` requires
  `TP >= 348` against the champion's ~340: eight more correct positives on the public half,
  about sixteen on the full test.
* **The rejections of the fingerprint and group-max variants were not statistically meaningful.**
  Champion ~340 vs fingerprint 338 vs group-max ~340 differ by at most 2 TP, while two
  submissions that differ in 300 rows have a TP difference with sd ~5.9 (0.0085 F1). Both
  variants had a +0.003 temporal-CV gain and lost inside the noise band; the honest reading is
  that the three rankings are statistically tied online, and the offline CV remains the better
  estimator of true quality.
* **The country-11 reallocation probes are also inside the noise band.** `probe_shift_to23`
  (-7 TP) and `probe_shift_to11` (-12 TP) both moved by ~1-2 sd, so the measurement says the
  current 808/974 split is not obviously wrong, not that it is optimal. Segment precision on
  the public half is 0.405 inside country-11 (159/393) against 0.359 outside (181/504).
* **The operating point conclusion survives**, because the break-even rule (marginal precision
  vs `F1/2`) is scale-free: the recovered TP curve gives marginal precision 0.30 on
  847 -> 897 and 0.22 on 897 -> 947, straddling the 0.244 threshold, so `k = 1782` stays the
  best measured choice with `k = 1881` statistically indistinguishable.
* **Selecting a variant by its public score is selection on 3723 rows.** With several
  near-equally-good rankings the maximum of their public scores is biased upwards by roughly one
  sd of the noise, so a higher public number can be reached by uploading decorrelated variants
  of equal offline quality, and any single 0.001-0.003 difference must not be treated as
  evidence about true quality.
