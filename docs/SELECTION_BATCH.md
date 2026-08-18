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
