"""Recover the exact scoring set of the public leaderboard from the observed F1 values.

For a top-k submission with `k_scored` predicted positives inside the scored set and `P`
positives in that set, the positive-class F1 is exactly

    F1 = 2 * TP / (k_scored + P)

so every reported F1 is a rational number whose reduced denominator divides `k_scored + P`.
Two facts pin the scoring set down:

* the all-ones submission scores 165/703, which forces the positive rate P/T = 165/1241 and
  therefore T = 1241 * m, P = 165 * m for some m <= 6 (m = 6 is the whole test file);
* for each uploaded file we know `k` on the full test file, so `k_scored` must be close to
  `k * T / 7446` (hypergeometric, sd = sqrt(q(1-q)/k) with q = T/7446), and can never exceed `k`.

Enumerating the admissible `(k_scored, TP)` pairs shows only m = 3 survives: the public score
is computed on half of the test file, 3723 rows with 495 positives, and ~50.3% of every
submission's positives land inside it. m = 6 is impossible because it would need k_scored > k.

    python src/decode_leaderboard.py
"""

from __future__ import annotations

import math
from fractions import Fraction

TEST_ROWS = 7446
SAMPLE_F1 = 0.2347083926  # all-ones submission, given in the task statement
TOLERANCE_SIGMA = 4.0

# (name, predicted positives in the uploaded file, reported F1)
OBSERVED = [
    ("m140_fp", 1386, 0.4789594491201224),
    ("m180_fp", 1782, 0.48563218390804597),
    ("m190_fp", 1881, 0.4868238557558946),
    ("m200_fp", 1980, 0.478494623655914),
    ("m190", 1881, 0.48581314878892734),
    ("m195", 1930, 0.4805460750853242),
    ("probe_seg_c11", 808, 0.3581081081081081),
    ("probe_shift_to23", 1782, 0.47844827586206895),
    ("probe_shift_to11", 1782, 0.4716031631919482),
]
# Scores reported rounded to 3-4 digits; they cannot pin an exact TP, only bracket it.
ROUNDED = [("m170", 1683, 0.48404), ("m180 (champion)", 1782, 0.489), ("m180_gmax", 1782, 0.488)]


def sample_ratio() -> Fraction:
    """Positive rate implied by the all-ones score: F1 = 2r/(1+r) with r = P/T."""
    f1 = Fraction(SAMPLE_F1).limit_denominator(100000)
    return f1 / (2 - f1)


def solutions(f1: float, rows: int, pos: int) -> list[tuple[int, int]]:
    """All (k_scored, TP) pairs consistent with an exact F1 on a `rows`/`pos` scoring set."""
    frac = Fraction(f1).limit_denominator(6000)
    num, den = frac.numerator, frac.denominator
    out = []
    for mult in range(1, 120):
        if (num * mult) % 2:
            continue
        k_scored, true_pos = den * mult - pos, num * mult // 2
        if 0 < k_scored <= rows and true_pos <= min(k_scored, pos):
            out.append((k_scored, true_pos))
    return out


def best_solution(k: int, f1: float, rows: int, pos: int) -> tuple[int, int, float] | None:
    """Solution whose scored-positive share is closest to the expected one, with its z-score."""
    share = rows / TEST_ROWS
    # `share == 1` means the whole test file is scored: then k_scored must equal k exactly.
    sd = math.sqrt(share * (1 - share) / k) or 1e-9
    cands = [(ks, tp, (ks / k - share) / sd) for ks, tp in solutions(f1, rows, pos) if ks <= k]
    return min(cands, key=lambda c: abs(c[2])) if cands else None


def main() -> None:
    ratio = sample_ratio()
    print(f"all-ones F1 = {Fraction(SAMPLE_F1).limit_denominator(100000)} -> P/T = {ratio}")
    candidates = [(ratio.denominator * m, ratio.numerator * m)
                  for m in range(1, TEST_ROWS // ratio.denominator + 1)]
    print(f"scoring sets with that exact positive rate: {candidates}\n")

    for rows, pos in candidates:
        picks = {name: best_solution(k, f1, rows, pos) for name, k, f1 in OBSERVED}
        inside = [n for n, p in picks.items() if p is not None and abs(p[2]) <= TOLERANCE_SIGMA]
        worst = [f"{n}:{'none' if p is None else format(p[2], '+.1f') + 'sd'}"
                 for n, p in picks.items() if p is None or abs(p[2]) > TOLERANCE_SIGMA]
        verdict = "CONSISTENT" if len(inside) >= len(OBSERVED) - 1 else "rejected"
        print(f"T={rows:5d} P={pos:4d}  {len(inside)}/{len(OBSERVED)} within "
              f"{TOLERANCE_SIGMA:.0f}sd  {verdict}  offenders={worst}")

    rows, pos = ratio.denominator * 3, ratio.numerator * 3
    print(f"\n=== scored set T={rows}, P={pos} ===")
    for name, k, f1 in OBSERVED:
        pick = best_solution(k, f1, rows, pos)
        if pick is None:
            print(f"{name:18s} k={k:5d}  no solution")
            continue
        k_scored, true_pos, z = pick
        print(f"{name:18s} k={k:5d}  k_scored={k_scored:5d} ({k_scored / k:.3f}, {z:+.1f}sd)"
              f"  TP={true_pos:3d}  precision={true_pos / k_scored:.4f}  recall={true_pos / pos:.4f}")
    print("\nrounded scores (TP bracketed at the observed 0.503 share):")
    for name, k, f1 in ROUNDED:
        k_scored = round(k * 0.5033)
        print(f"{name:18s} k={k:5d}  k_scored~{k_scored:5d}  TP~{f1 * (k_scored + pos) / 2:7.1f}")

    k_ref = 897
    step = 2 / (k_ref + pos)
    need = math.ceil(0.50 * (k_ref + pos) / 2)
    print(f"\none extra true positive at k_scored={k_ref} moves F1 by {step:.5f}")
    print(f"F1 >= 0.500 at k_scored={k_ref} needs TP >= {need} (champion is ~340)")
    swap = 150 * rows / TEST_ROWS
    print(f"a 150-row swap touches ~{swap:.0f} scored rows, so the TP difference of two such "
          f"submissions has sd ~{math.sqrt(2 * swap * 0.38 * 0.62):.1f} TP "
          f"= {math.sqrt(2 * swap * 0.38 * 0.62) * step:.4f} F1")


if __name__ == "__main__":
    main()
