# Cached selection batch

Эта пачка построена только из сохранённых test-side вероятностей:

- `artifacts/probs_test.npz` — legacy equal4;
- `artifacts/probs_test_reproduced.npz` — reporter-fingerprint equal4.

Новых обучений не выполнялось. Для всех вариантов использованы стабильная сортировка
по убыванию score (`argsort(kind="stable")`), ровно заданное число единиц и исходный
порядок строк `test.csv`. Все файлы имеют колонки `claim_id,is_valid`, 7446 строк и
бинарные метки.

В предыдущем CV-анализе различия между этими рангами были малы относительно
наблюдаемого шума: варианты статистически связаны с чемпионом по четырём
хронологическим фолдам. Они различаются главным образом составом верхних строк,
что и проверяется размером пересечения с champion top-k.

## Результаты

`CV F1` — средний `f1_m1.8` из cached blend-анализа предыдущего раунда.
Пересечение указано с `submissions/submission_equal4_m180.csv` (из 1782 строк).

| Файл | Рецепт | k | CV F1@m1.8 | Пересечение | SHA256 |
|---|---|---:|---:|---:|---|
| `cand_seed42.csv` | legacy equal4, seed 42 | 1782 | 0.42215 | 1747 | `4824ba9eea567f0fe1ab1e4e76b945006dbf530b2128f03ac29c46b144904b5f` |
| `cand_wo_cat_recent.csv` | legacy без `cat_recent` | 1782 | 0.42158 | 1716 | `1584284196e9e078c24a32f2e120ca2b504bcc362a239d7896b347cce5a040ad` |
| `cand_rank_legacy_fp.csv` | rank-average legacy + fingerprint | 1782 | 0.42024 | 1738 | `e40e692399a963f18ae235e0ac1d113cb72b3ecd458edeeefd2ee882bf31969e` |
| `cand_wo_cat_all.csv` | legacy без `cat_all` | 1782 | 0.42003 | 1706 | `958b9ada35c35bfb5ea5b44118a11d990cf9e579baffc7727c2105d50606052d` |
| `cand_legacy_rank.csv` | rank-average четырёх legacy-компонент | 1782 | 0.42000 | 1747 | `bf764b6abee80ab16aff6efec8905a93cdc4497e3fbd6e0abdd66dd621dbf357` |
| `cand_seeds2026_777.csv` | legacy equal4, seeds 2026 и 777 | 1782 | 0.41932 | 1766 | `ddaddbf02d3de9c71f23095001be40e9406e05f9c66728524125eeb97f18650f` |
| `cand_rank_all3.csv` | rank-average legacy + fingerprint + gmax | 1782 | 0.41865 | 1748 | `1ae3682a6ddc46aa660c0c497a75ef2806facf15bc1d60fca7d5ec7a5da693e8` |
| `cand_seed42_m190.csv` | то же ranking, seed 42 | 1881 | 0.42215 | 1770 | `b39566e895b3f30e6188c6fd2baadca306a2f2a32e713b3777dde548a37cde1b` |
| `cand_wo_cat_recent_m190.csv` | то же ranking, legacy без `cat_recent` | 1881 | 0.42158 | 1755 | `8ace72eebce97789d5f33fa6ad2949328ed51a9b7518bbf44d25f4c2729ec261` |
| `cand_rank_legacy_fp_m190.csv` | то же ranking, rank-average legacy + fingerprint | 1881 | 0.42024 | 1772 | `342330e8a6e5a7efb6d3694c847e3337246d89275395022340506c07effba150` |

## Воспроизведение

```bash
python src/make_selection_batch.py
```

Скрипт проверяет количество строк, число позитивов, порядок `claim_id`, бинарность
меток и печатает SHA256 каждого результата. Исходный `submission.csv` скрипт не
изменяет.

## Измерения публичного leaderboard

Эти значения получены на публичной половине теста и считаются
авторитетными для данного раунда. Здесь `T=3723`, `P=495`, а
`F1 = 2*TP/(k_scored+495)`. Результаты относятся к уже загруженным файлам.

| Файл | k | Exact F1 | k_scored | Public TP |
|---|---:|---:|---:|---:|
| `cand_rank_all3` | 1782 | 0.49246231155778897 | 898 | 343 |
| `cand_seeds2026_777` | 1782 | 0.4921090387374462 | 899 | 343 |
| `cand_rank_legacy_fp` | 1782 | 0.4878048780487805 | 899 | 340 |
| `submission_equal4_m180` | 1782 | ~0.489 | ~897 | ~340 |
| `cand_seed42` | 1782 | 0.4860215053763441 | 900 | 339 |
| `cand_wo_cat_all` | 1782 | 0.4794816414686825 | 894 | 333 |
| `cand_wo_cat_recent` | 1782 | 0.4789135096497498 | 904 | 335 |
| `cand_seed42_m190` | 1881 | 0.48261474269819193 | 943 | 347 |
| `cand_rank_legacy_fp_m190` | 1881 | 0.48476454293628807 | 949 | 350 |

На измеренных загрузках лидирует направление rank-average: `cand_rank_all3`
получил 343 public TP. Увеличение k до 1881 не стало улучшением:
маржинальная точность расширения остаётся ниже break-even. Вывод ограничен
публичной половиной теста и не является оценкой всего тестового набора.

## Batch 2: consensus-кандидаты

Все пять новых файлов используют `k=1782`. Пересечения указаны отдельно с
champion `submission_equal4_m180.csv` и с уже лучшим `cand_rank_all3.csv`.
Все новые файлы прошли проверки размера, схемы, порядка строк и бинарности.

| Файл | Рецепт | Champion overlap | `cand_rank_all3` overlap | SHA256 |
|---|---|---:|---:|---|
| `cand_rank_top5.csv` | rank-average: legacy, fingerprint, gmax, seed 42, seeds 2026+777 | 1758 | 1772 | `24843b8873e16f6078eeda8b9288caf8580af45a2ba135941292e252ba0bdd58` |
| `cand_vote_top4.csv` | top-k vote legacy/fingerprint/gmax/seeds 2026+777, tie-break rank-average | 1768 | 1761 | `b5390e6908fb957059cfab9533d29a7d9b4b62d21bfafadeb2f785695a8436ce` |
| `cand_gmax_hi.csv` | rank-average legacy + fingerprint + gmax, `wc=0.25`, `wo=0.15` | 1748 | 1779 | `63c858235af0d83aee726b8573d97f7e942818bc84a3a4061b3b340c5f5e2cc3` |
| `cand_gmax_lo.csv` | rank-average legacy + fingerprint + gmax, `wc=0.08`, `wo=0.05` | 1749 | 1778 | `117345c646c880099289284dfb94720127de9d020f11ff0b253a24db39218973` |
| `cand_rank_wide.csv` | rank-average пяти base scores плюс legacy equal5 с `lgb_strong` | 1762 | 1767 | `e17411b3059528556761b5ab7b3009cff452ab7fc09a8aa0b323082d05cb83ea` |

Ни один из пяти новых файлов не совпадает побайтно с ранее загруженным
кандидатом.

Batch 2 воспроизводится командой:

```bash
python src/make_batch2.py
```
