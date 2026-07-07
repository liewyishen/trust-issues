# Data Decisions

Record of data-quality investigations that changed a contract in
`src/data_validation.py` (or would otherwise be non-obvious from reading the
code alone). Append new entries below; don't overwrite prior ones.

---

## dti_n > 100 的 495 行:真实极端客户,非脏值

**Date:** investigation run against the full real CSV, first time
`validate_loan_data()` was wired into `load_raw()` (see `src/data_loader.py`).

**Finding:** the original schema contract `(0 ≤ dti_n ≤ 100) OR (dti_n == 999)`
rejected 495 real rows with `dti_n` strictly between 100.04 and 991.57 --
neither in the real band nor the known missing-value sentinel.

**Investigation, four pieces of evidence:**
1. **违约率反证 (default-rate reversal):** the 495 rows have a 27.07% default
   rate vs. 19.98% overall. If these were a decimal-shift artifact (i.e. the
   true value is `dti_n / 100`), the corrected values would be the *lowest*
   leverage in the dataset (median ~1.5% DTI), predicting a *below*-average
   default rate -- the opposite of what's observed. This rules out the
   decimal-shift hypothesis.
2. **时间断层 (temporal cliff):** 0 rows in Train (2007-2014), 7 in 2015, 488
   in 2016-2018. The near-total absence before 2016 and concentration after
   points to a change in LendingClub's DTI computation/reporting methodology
   around 2015-2016, not to random data corruption.
3. **非哨兵 (not sentinel/encoding residue):** 483 of 495 values are distinct,
   continuously distributed from 100.04 to 991.57 at the same two-decimal
   precision as ordinary DTI. Sentinels cluster on a handful of reused, exact
   values (like -1 or 9999) -- this population doesn't.
4. **特征画像 (feature profile):** median revenue $103k vs. $65k overall,
   median loan_amnt $19.2k vs. $12k overall, FICO not depressed. Consistent
   with a real subpopulation of high-income, high-loan-amount borrowers whose
   reported DTI genuinely exceeds 100%, not with a noisy/garbage profile.

**Verdict:** real extreme customers, most likely reflecting a genuine 2016+
change in how DTI was computed or reported upstream -- not sentinel residue,
not a decimal-point error.

**Disposition:** `DTI_MAX_REAL` widened from 100 to 1000 in
`src/data_validation.py` (covers the observed real ceiling of 991.57 with a
small margin). `DTI_SENTINEL = 999` is kept as a separate, explicit OR branch
in the schema check even though it's now numerically redundant (999 ≤ 1000)
-- it remains a distinct missing-value sentinel semantically, not a real DTI
value, and merging it away would erase that meaning from the code. Values
still keep no evidentiary basis beyond 1000 (e.g. a stray 9999) continue to
be rejected.

**待办 / TODO:**
- `pipelines/drift_check.py` (not yet built) must monitor `dti_n`'s
  distribution by `issue_year`, not just its marginal range -- the 2016+
  regime is invisible to a check that only looks at the overall histogram,
  since Train never sees it and Test/2018-holdout do. This is a live
  train/serve distribution-shift signal, not just a one-time data-hygiene
  fix.

---

## MLflow tracking backend: file store → SQLite

**Finding:** `src/train.py` originally pinned MLflow's tracking URI to a
plain filesystem store (`file://.../mlruns`) and had to set
`MLFLOW_ALLOW_FILE_STORE=true` to opt back into it, because the installed
`mlflow>=3.14` puts that backend into "maintenance mode" and refuses to
create a new file-based run store otherwise. That flag was a workaround, not
the recommended path, and it meant `mlflow ui` still failed unless the same
env var was set by hand on the command line.

**Disposition:** migrated the tracking URI to a local SQLite database
(`sqlite:///{PROJECT_ROOT}/mlflow.db`, absolute path so it resolves to the
same file regardless of invocation directory). `MLFLOW_ALLOW_FILE_STORE` is
no longer set anywhere in this codebase. `mlflow ui` now starts with:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

no extra flags or environment variables required.

**Note on `./mlruns`:** it is not fully obsolete. MLflow's artifact store
(where `mlflow.log_artifact()` actually copies the joblib model file) is a
separate concept from the tracking backend and still defaults to a local
`./mlruns` directory regardless of whether tracking points at a file store
or a database -- only run *metadata* (params/metrics/tags) moved into
`mlflow.db`. Both `mlflow.db` and `mlruns/` are gitignored.

**待办 / TODO:** none currently; revisit if/when `pipelines/training_flow.py`
needs a shared (non-local) MLflow backend for multi-machine runs.

---

## MLflow TODO: calibrate.py / fairness.py 指标未接入 mlflow

**待办 / TODO:** calibrate.py 的 Brier/mean_pred 和 fairness/evaluate 的关键
指标目前只打印到终端,未接入 mlflow。待 pipeline(Metaflow)阶段,把训练 AUC
+ 校准 Brier + 评估指标统一记录到单次 mlflow run,形成一个模型的完整实验
档案。

---

## 执行 fairness 结论:生产模型移除 addr_state

**审计证据摘要**(详见本文件上方 fairness 相关记录及 `src/fairness.py`
module docstring):三层审计确认 `addr_state` 是数字红线捷径,而非真实经济
差异的代理。Layer 3 消融(阈值 0.22):Mississippi 好客户 Equal Opportunity
ratio,含州时 ~0.734~0.745,去州后 ~0.988~0.990;去州的 test AUC 代价仅
-0.0035~-0.0036(0.6689/0.6690 → 0.6654)。结论:生产模型移除 `addr_state`
带来的 fairness 收益远大于其预测力贡献。

**实现方式:配置开关,非硬删。** `src/features.py` 新增模块级开关
`INCLUDE_ADDR_STATE = False`(默认关闭,执行审计结论),`CATEGORICAL` 由
`build_categorical(INCLUDE_ADDR_STATE)` 动态构成;`addr_state` 的列定义、
`emp_order` 等特征工程逻辑原样保留。选择开关而非直接从代码里删掉
`addr_state`,是为了保留复现能力:`build_categorical(True)` 仍能一键重建
"含州"特征集,让 `fairness.py` 的 Layer 3 消融可以随时重新对比、重新验证,
而不是只留一份写死在文档里的历史结论。`src/fairness.py` 的 Layer 3(
`audit_layer3_ablation`)因此被重构为完全自包含:不再读取
`features.py` 的(随开关变化的)`FEATURES`/`CATEGORICAL`,也不再依赖任何
已加载的生产模型作为"含州"一侧的对比基准,而是用本模块自己的
`FEATURES_WITH_STATE`/`FEATURES_NOSTATE` 常量,每次都从头训练两个变体
——不论当前生产实际部署的是哪一个。

**重训后的新数字**(真实数据,`INCLUDE_ADDR_STATE=False`):

```
calibrate_model():
  Raw (uncalibrated):     Brier=0.1717  mean_pred=0.1705  AUC=0.6660
  Calibrated (isotonic):  Brier=0.1692  mean_pred=0.1915  AUC=0.6654

run_evaluation():
  Threshold chosen on VAL: 0.25
  Test profit @ 0.25:        -$288,367,478
  Naive 0.50 profit:         -$473,187,818
  Improvement over naive:    +$184,820,340
  Approval rate on test:      78.0%
  Bad rate among approved:    18.9%
  Bad rate among rejected:    38.5%

run_fairness_audit() Layer 1 (threshold=0.26): 全部州 verdict=clear,
无一州 CI 完全低于 0.80(此前含州版本 MS 在更紧阈值下会被 Layer 2 揭示出
持续低于 0.80 的问题;去州后 MS 在阈值 0.26 的 EO ratio 已升至 ~0.994)。
```

**对照含州版本**(移除前的历史数字,供参考):test AUC 0.6689,阈值 0.26,
改善 ≈ +$190.6M,批准率 80.3%,approved 坏账率 19.2%,rejected 坏账率
39.6%。AUC 略降(-0.0035 附近)、改善金额略降(~$190.6M → ~$184.8M)、批
准率略降(80.3% → 78.0%)都是预期且可接受的代价——这就是消除数字红线
风险的价格,不是模型变差了。

**待办 / TODO:** 无。若未来重新评估是否恢复 `addr_state`,应重新跑一次
`fairness.run_fairness_audit()` 而非直接假设历史数字仍然成立。
