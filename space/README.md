---
title: BreachLab Guard
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Prompt-injection and jailbreak guard for the BreachLab game
---

# BreachLab guard service

CPU inference service for the BreachLab game guard: a Qwen3-0.6B LoRA fine-tune exported to ONNX, one forward
pass, 4-way verbalizer softmax (`safe`, `injection`, `jailbreak`, `exfiltration`), temperature-calibrated.
**Research demo, not a production safety system.** Weights are pulled from the Hub model repo
`Pseudo-Harsh/breachlab-guard-v1` at boot.

## API (frozen, `docs/CONTRACTS.md` section 1)

`POST /score`

```json
{"text": "ignore all previous instructions", "kind": "input", "alias": "champion"}
```

returns

```json
{"label": "UNSAFE", "category": "injection", "p_unsafe": 0.97, "confidence": 0.97,
 "version": "sft-0.6b-s0", "latency_ms": 640.2}
```

- `alias`: `champion` (SFT, served at Levels 4-5), `challenger` (SFT+DPO, shadow), `base` (untuned Qwen3-0.6B,
  Level 3). An unknown alias falls back to `champion`.
- `kind`: `input` (user message) or `output` (assistant reply).
- `confidence` always equals `p_unsafe`. `latency_ms` is model time only (queueing excluded).
- 503 while the models are still loading or when inference fails (the gateway must fail closed).

`GET /health` returns `{"ok": true, "models": {"champion": "...", "challenger": "...|null", "base": "...|null"}}`
(503 with `"ok": false` until the champion is loaded). `GET /info` (not part of the contract) reports the ONNX
variant, the onnxruntime version and the cold-start timings.

## Configuration (environment variables; Space variables / secrets on Hugging Face)

| name | default | meaning |
|---|---|---|
| `GUARD_MODEL_REPO` | `Pseudo-Harsh/breachlab-guard-v1` | Hub repo with `champion/`, `challenger/`, `base/` |
| `GUARD_VARIANT` | `fp32` | `kv` (prefix-KV fp32, same maths as fp32, about 2x faster), `fp32`, `w8` (weight-only int8, slower) or `dyn_int8` (rejected, quality). An alias that lacks the variant falls back to fp32 |
| `GUARD_ALIASES` | `champion,challenger,base` | aliases to load (`champion` mandatory); `champion,base` halves memory and download |
| `GUARD_THREADS` | cgroup CPU quota | ONNX Runtime intra-op threads |
| `GUARD_MAX_TOKENS` | `configs/serve.yaml` (512) | message token limit override (quality cost not free, see docs/lanes/B2-truncation-quality.json) |
| `GUARD_TOKEN` (secret) | none | when set, `POST /score` requires `Authorization: Bearer <token>` (the gateway sends its own `GUARD_TOKEN`) |
| `GUARD_BOOT_SYNC` | off | `1`: load the models before the port opens (hosts that throttle idle CPU) |
| `GUARD_CACHE_DIR` | `/tmp/hf-guard` | where the weights are downloaded (needs the full model size on disk or RAM) |
| `HF_TOKEN` (secret) | none | fine-grained read-only token for the private model repo |

Hosting options and exact commands: `docs/deploy-guard.md`. The YAML front matter above is only read by Hugging Face.

## Repo layout

Built by `python -m breachlab.serve.space_build --stage-only DIR` (or `deploy` for a Space) from the BreachLab
repository: this folder plus byte-identical copies of the runtime modules (`breachlab/...`) and
`configs/serve.yaml`, so training and serving share one prompt and one scoring implementation.
