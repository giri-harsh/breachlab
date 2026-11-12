# Retraining loop (designed, not scheduled)

`retrain.yml` is `workflow_dispatch` only (PRD 10). The loop, with what is implemented today:

1. **Capture** every message in `attempts` (implemented: gateway). Canary sessions are flagged and excluded.
2. **Label** (`labels` table): leaked -> UNSAFE conf 1.0 (implemented, source=`leak`); blocked -> judge label
   (Gemini judge, `eval/external.judge_batch`); passed and no leak -> benign candidate, judge on a 20% sample;
   judge confidence < 0.7 or judge != guard -> `/admin/review` queue (implemented: queue + human labels).
   Track Cohen's kappa judge vs human on the reviewed rows.
3. **Build** a new dataset version: PII scrub -> dedup -> carve `fresh_test` -> merge -> validate -> `data-vN`
   (`data/build.py`; the live-data step reads labelled attempts).
4. **Train** (`retrain.yml` -> `train/kaggle_job.py` or Modal): SFT -> mining -> DPO -> calibration. Results land
   in MLflow with GPU-seconds and USD; the new version gets alias `challenger`.
5. **Evaluate + gate**: `scripts/eval_gate.py` runs `configs/gate.yaml` on gold_test and fresh_test. Failing
   versions stay in MLflow with the reason; nothing ships.
6. **Shadow** for 48 h: the gateway scores 100% of live traffic with the challenger and records
   `shadow_verdict`; compare agreement and shadow ASR on leaked attempts.
7. **Promote**: `promote.yml` moves alias `champion` (old champion kept as `previous`). The gateway polls the
   alias every 30 s and hot-swaps.
8. **Rollback**: `rollback.yml` moves `champion` back to `previous`. Measured alias move: ~1.2 s (see
   docs/progress.md).

First cycle starts once at least 1,000 labelled live attempts exist.
