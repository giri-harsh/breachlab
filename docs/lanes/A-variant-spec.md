# Lane A: winning ONNX variant spec for the guard (for Lane B)

Date 2026-10-30. Evidence: `data/eval/exportA/summary.json` (+ `export3.json`, `gold300_*_logits.npy`) from Kaggle CPU
kernel `giriharsh/breachlab-exporta-sft17` (v1, COMPLETE), and `data/eval/export2/parity.json` (DPO adapter,
kernel `giriharsh/breachlab-export2-06b-s0`). Both in DVC (`data/eval.dvc`). Reproduce the tables with
`uv run python scripts/run_parity.py --export3 data/eval/exportA --out data/eval/exportA/summary.json`.

## Decision

**SHIP: ONNX fp32 export of the SFT adapter (`sft-0.6b-s0`), file `guard-fp32.onnx` + `guard-fp32.onnx.data`
(about 2.4 GB), tokenizer `tokenizer.json`.**

IMPORTANT for Lane B: kernel `export2` exported the DPO adapter (`dpo-0.6b-s0`). Per ADR-0036 the shipping model is the
**SFT-only** adapter, so the shipping export must be re-done for `sft-0.6b-s0` (adapter at
`giriharsh/breachlab-sft-06b-s0` -> `out/sft-0.6b-s0/adapter`). The numbers below are for exactly that SFT adapter
(kernel exporta, `which=sft-0.6b-s0`), so the parity result transfers; Lane B's own `breachlab-export3-sft` kernel
(same adapter) can be used for the files.

Calibration to ship with it (fitted on val for the SFT adapter, FPR<=2%): temperature 1.4115724021135456,
threshold 0.06332372686360212 (`data/eval/sft06b/results.json`, `data/eval/dpo_results.json` key `sft`).

## Parity vs torch (first 300 gold_test rows: 214 benign, 86 attacks; the rows the kernel scored)

| variant | AUROC torch / onnx | FPR at tuned thr torch / onnx | ASR torch / onnx | label agreement | max abs logit diff | verdict |
|---|---|---|---|---|---|---|
| fp32 | 0.9926 / 0.9926 | 2.34% / 2.34% | 4.65% / 4.65% | 100% | 0.039 | OK (winner) |
| dynamic int8 | 0.9926 / 0.9787 | 2.34% / 24.30% | 4.65% / 1.16% | 83.3% | 8.1 | FAILS (over-blocks benign) |
| weight-only int8 (w8, DPO adapter, export2) | 0.9937 / 0.9937 | 2.34% / 2.34% | 3.49% / 3.49% | 100% | 0.196 | OK but slower than fp32 |

Rule (src/breachlab/eval/parity.py, ADR draft): AUROC drop <= 0.003, FPR rise <= 0.5 pp, ASR rise <= 1.0 pp.
Only 300 rows could be compared: the export kernel scores `gold[:300]` (pipeline.run_export); the full 1,500-row
parity needs a kernel change (REQUEST in the lane report). The 1.7B shows the same pattern (fp32 parity OK, dynamic
int8 collapses to AUROC 0.4466).

## Kaggle CPU latency (4 vCPU, 4 threads, 200 requests after 10 warm-up, message capped at 500 chars, concurrency 1)

| model / variant | 512-token cap p50 / p95 ms | 128-token cap p50 / p95 ms |
|---|---|---|
| 0.6B SFT fp32 (winner) | 733 / 3112 | 731 / 1514 |
| 0.6B SFT dynamic int8 (fails parity) | 448 / 1711 | 442 / 918 |
| 1.7B SFT fp32 | 1998 / 7769 | 1929 / 3958 |
| 1.7B SFT dynamic int8 (fails parity) | 1287 / 4896 | 1262 / 2658 |
| 0.6B DPO fp32 (export2, earlier machine) | 643 / 2284 | 653 / 1331 |
| 0.6B DPO w8 (export2) | 1402 / 3286 | 1380 / 2106 |

Run-to-run machine noise is about +-15% (same fp32 graph: 643 vs 733 ms p50). The 300 ms p95 gate is not met by any
variant that keeps parity; the HF Space (2 vCPU) will be slower than this 4-vCPU kernel. p50 is flat across the 512 and
128 caps (fixed ~115-token prompt cost); only the p95 tail moves with the cap. Untested options for Lane B: prefix KV
reuse for the fixed template, session `intra_op_num_threads` = vCPUs, and a 128-token cap (its quality loss on gold
is not measured; it needs a GPU/CPU scoring run).

## 0.6B vs 1.7B (same kernel, same fp32 variant)

0.6B SFT keeps 99.6% of the 1.7B SFT F1 on gold_test (calibrated on val, seed 0) at 2.7x lower p50 and 2.5x lower p95 latency.
