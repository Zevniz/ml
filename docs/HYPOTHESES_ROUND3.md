# Раунд 3: reporter fingerprint и абляции

Раунд 3 проверял, можно ли восстановить часть недоступной reporter history по
стабильным атрибутам профиля, а также проверял несколько расширений этого ключа.
Протокол был leakage-safe и воспроизводимым:

* `src/features_v2.py` содержит fingerprint и backward-looking history features;
* `src/exp_features.py` запускает абляции по вариантам признаков и моделям;
* `src/eval_utils.py` считает ROC AUC, PR AUC и top-k F1;
* `src/compare_cv.py` сравнивает сохранённые baseline и новые вероятности;
* проверка проводилась на 4 chronological folds, по 7 446 строк, с seed'ами
  `42, 2026, 777`;
* для полного ансамбля использовались равные веса четырёх компонент;
* target encoding считается OOF внутри train prefix с `alpha=15`, а validation/test
  получают mapping, обученный только на prefix.

## Результаты

Reporter fingerprint — это 10 стабильных атрибутов профиля, соединённых в один ключ:
`registered_year`, `age_bucket`, `sex`, `has_avatar`, `has_school`,
`has_university`, `is_private_profile`, `registered_phone_country_id`,
`mobile_phone_country_id`, `profile_country_id`. Plain fingerprint TE с prefix count
принят только для LightGBM.

| Вариант | m=1.6 | m=1.7 | m=1.8 | m=1.9 | m=2.0 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.4183 | 0.4190 | 0.4223 | 0.4230 | 0.4212 |
| Reporter fingerprint TE | 0.4190 | 0.4232 | 0.4252 | 0.4252 | 0.4231 |

На shipped operating point `m=1.8` получено `0.4223 -> 0.4252` (`+0.0029`).
Per-fold ROC AUC изменился с `0.7940` до `0.7952`, PR AUC — с `0.4121` до
`0.4145`; fold 1 немного ухудшился по PR AUC (`0.4612 -> 0.4554`), тогда как
folds 4/3/2 улучшились по обеим метрикам. `src/rate_curve.py` сохранил оптимум
`m=1.8`, то есть `RATE_MULTIPLIER=1.8` и `k=1782` для блока из 7 446 строк.

LightGBM ablation:

* base: PR AUC `0.4052`, F1 `0.4162`;
* plain fpte: PR AUC `0.4126`, F1 `0.4208`;
* fpte_a40: PR AUC `0.4130`, F1 `0.4192`;
* `fp11te`, `fpte_type`, `fpte_reason`, `fpte_nocount`, `fpte_a5` — не лучше plain
  fpte и отклонены;
* backward-looking `hist`: PR AUC `0.4040`, neutral-to-negative, отклонён.

CatBoost ablation (seed 42, 4 folds) дал PR AUC `0.4016` для base против `0.3971`
для fpte; на fold 1 PR AUC упал с `0.4506` до `0.4250`. Поэтому
`reporter_fp_te` и `reporter_fp_history_count` исключаются из CatBoost feature set:
`base_columns` не включает id-based `_te`/`_history_count` колонки в `cat_features`.

## Принятые команды

```bash
python src/exp_features.py --variant base --model lgb --seeds 42 --folds 1
python src/cv_probs.py --folds 4 --seeds 42,2026,777 2>&1 | tee artifacts/cv_run_fp.log
python src/compare_cv.py artifacts/baseline artifacts
python src/rate_curve.py
```

Для полного ансамбля и финального файла:

```bash
python solution.py 2>&1 | tee artifacts/solution_final_fp.log
cp submission.csv submissions/submission_equal4_m180_fp.csv
```
