# Resume bullets: BreachLab

**One line:** BreachLab: a $0, end-to-end MLOps project: a fine-tuned Qwen3-0.6B prompt-injection guard (int4 ONNX on CPU) defending a live web game, with data versioning, an eval gate, a model registry with rollback and a canary.

## Bullets
- **Engineered** a prompt-injection guard by SFT of Qwen3-0.6B with LoRA on 19,330 training rows (27,977 total, 1,500-row held-out gold set): attack success rate fell from **100% (zero-shot) to 0.6%** at 1.6% FPR (AUROC 0.999), and beat DeBERTa (10.4% ASR / 14.2% FPR), Llama Prompt-Guard-2 (51.6% ASR) and a Gemini judge (12.8% ASR / 2.8% FPR) on the same set; temperature scaling cut ECE from 0.023 to 0.009.
- **Pioneered** a measured fix for a hidden failure mode: found that the guard blocked **54.3%** of 300 held-out short benign openers (96% for Hindi), built a 2,172-row hard-benign en/hi/hinglish dataset (data-v2, DVC-versioned), retrained on a Kaggle T4 at $0, and cut opener FPR to **6.3%** (Hindi 15.0%) while gold ASR improved from 0.59% to **0.44%** and FPR from 2.08% to 1.83% (AUROC 0.9998), all behind four promotion gates defined before the run.
- **Spearheaded** a CPU-only, $0 serving stack: int4 weight-only ONNX with prefix-KV reuse (433 MB vs 2,384 MB fp32; dynamic int8 had 24.3% FPR vs 2.3%) on a 1 vCPU Vercel function, with MLflow champion/challenger/previous aliases (promote 3.0 s, rollback 4.9 s median), a fail-closed gateway and a cold-start-aware UI; measured **p50 722 ms / p95 2,063 ms** model time warm for the 0.6B, about 13 s guard cold start (the 300 ms target is not met, and documented as such).
- **Shipped** a red-team harness and live game (Next.js, FastAPI, Neon Postgres) with 5 bosses plus a daily boss, a deterministic multi-encoding leak detector that labels every attempt for free, **216 template + 110 fresh attacks per boss** (bosses 4 and 5: 0 of 326 leaks) and 573 automated tests at 96% coverage plus mocked and production Playwright e2e at 390 and 1440 px.
- **Distilled** the 0.6B guard into a 136 MB DistilBERT int8 student (KD on teacher soft labels, Kaggle T4, $0) and served a **student-first cascade** that escalates only the uncertain band to the 0.6B (4.8% of production requests): gold ASR 0.44% / FPR 1.1% unchanged, student 10-40 ms per message; moving the gateway next to the database and dropping a duplicate shadow call cut the full request p50/p95 from 5.1/6.4 s to 2.7/4.0 s (gateway time 0.5/1.1 s). Reported the cost honestly: on a 934-row out-of-distribution set the student's ASR is 13.4% vs 6.6% for the 0.6B and the cascade 9.5%.
- **Audited** my own evaluation: built a 934-row out-of-distribution test set (public MIT/Apache/CC0 sets, different-family author and generator, Hinglish, MinHash leakage checks at Jaccard 0.7) and an LLM-consensus review of the gold labels (361 rows, 3 judges, kappa 0.83, estimated label noise 3.3% [1.8, 6.0]), then measured cost at $0.11 per 1,000 requests at list prices (actual bill $0).

## 30-second pitch
"I built BreachLab, a small web game where you try to trick an AI vault into leaking a secret, and I used it as an end-to-end MLOps project. The defence is a Qwen3-0.6B guard I fine-tuned with LoRA: it takes attack success from 100% zero-shot to under 1% with about 2% false positives, served as an int4 ONNX model on a free CPU function. The interesting part is what I found when I tested it honestly: it flagged more than half of short, harmless messages, especially Hindi ones. I wrote targeted data, retrained, and brought that from 54% to 6% without hurting the main metrics, with the gates, registry and rollback all in place. It's not perfect: latency misses my 300 ms target (a distilled student plus cascade gets most traffic under it, the rest still takes about 2 s), my main eval set is in-distribution (the out-of-distribution one is worse: 6.6% ASR for the 0.6B), and 0% on scripted attacks doesn't mean unbreakable, and I say so in the README."

## Where every number comes from
| claim | source |
|---|---|
| zero-shot ASR 100%, fine-tuned 0.6% / FPR 1.6% / AUROC 0.999, baselines, ECE 0.0234 to 0.0087 (`docs/lanes/A3-int4-calibration.json`), 19,330 / 27,977 rows, 1,500 gold | `reports/eval-v1.md`, `web/public/scorecard.json`, `PROJECT_SUMMARY.md` (data v1 split counts) |
| opener FPR 54.3% to 6.3%, Hindi 96% to 15.0%, gold ASR 0.59% to 0.44%, FPR 2.08% to 1.83%, AUROC 0.9998 | `reports/eval-v2.md`, `docs/lanes/T-guard-v2-gates.json`, `docs/lanes/S16-opener-fpr.json` |
| 2,172 hard-benign rows, train 21,242 | `data/processed/v2/v2_stats.json` (DVC), `docs/data/DATASET_CARD_v2.md`, ADR-0094 |
| int4 433 MB vs fp32 2,384 MB, int8 FPR 24.3% vs int4 2.3% | `reports/eval-v1.md` (export and int4 sections), `PROJECT_SUMMARY.md` |
| promote 3.0 s, rollback 4.9 s (medians) | `docs/lanes/P-rollback-timing.json` |
| p50 722 ms, p95 2,063 ms (v2, warm, c1, 120 rows) | `docs/lanes/T-v2-latency.json`; cold start 12.8 s: `reports/eval-v1.md` (Cold start) |
| 216 + 110 attacks per boss, 0/326 on bosses 4 and 5 | `docs/lanes/G-difficulty-prod.json`, ADR-0102 |
| 573 tests, 96% coverage | `uv run pytest` summary (2026-11-12 gate) |
| student vs 0.6B vs cascade table, 4.8% escalation, full-request p50/p95 2.7/4.0 s client, 0.53/1.14 s server | `reports/eval-v2.md` (cascade section), `docs/lanes/D-cascade-v2.json`, `docs/lanes/D-full-request-cascade.json`, ADR-0105 |
| OOD set 934 rows, 0.6B ASR 6.6% / FPR 17.3%, student 13.4%, cascade 9.5% | `docs/lanes/D-cascade-v2.json`, `docs/data/OOD_CARD.md`, ADR-0106 |
| gold review kappa 0.828, noise 3.3% [1.8, 6.0], $0.1124 per 1k requests | `docs/gold-review.md`, `docs/lanes/C-cost.md`, ADR-0106/0107 |
