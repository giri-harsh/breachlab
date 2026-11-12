"""Body of the Kaggle CPU export kernel (embedded by scripts/run_kaggle_export.py after a prelude).

For each (which, base_model) in SPECS: merge the SFT adapter, export ONNX fp32 (external data if > 2 GB) and dynamic
int8, score the first 300 gold rows with every variant (parity input) and run the PRD 7.3 latency harness on this
CPU. Several models in one kernel means identical hardware for the 0.6B-vs-1.7B speed ratio. Results are written
after every step to /kaggle/working/out/export3.json plus gold300_<which>_<variant>_logits.npy; failures are
recorded per variant and never abort the run. Standalone because the code dataset on Kaggle is a frozen copy and
the owning lane's pipeline.run_export is hard-wired to one adapter.
"""

import gc
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

SPECS: list[tuple[str, str]] = list(globals().get("SPECS") or [])
REQUESTS: int = int(globals().get("REQUESTS") or 200)
KEEP_ONNX: list[str] = list(globals().get("KEEP_ONNX") or [])
OUT = Path("/kaggle/working/out")


def save(res: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "export3.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")


def find_adapter(which: str) -> Path:
    return next(p for p in Path("/kaggle/input").rglob("adapter") if p.parent.name == which)


def export_onnx(base_model: str, adapter: Path, ids: list[int], d: Path) -> dict[str, float]:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    secs: dict[str, float] = {}
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(adapter))
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
    lm = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload().eval()

    class Slim(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.body = lm.model
            self.register_buffer("head", lm.lm_head.weight[ids].detach().clone())

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            pos = (attention_mask.cumsum(-1) - 1).clamp(min=0)
            h = self.body(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=pos
            ).last_hidden_state
            return h[:, -1, :] @ self.head.T

    slim = Slim().eval()
    sample = torch.tensor(
        [tok.encode("<|im_start|>user\nhello<|im_end|>", add_special_tokens=False)]
    )
    tok.save_pretrained(d)
    torch.onnx.export(
        slim,
        (sample, torch.ones_like(sample)),
        str(d / "guard-fp32.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits4"],
        dynamic_axes={
            "input_ids": {0: "b", 1: "s"},
            "attention_mask": {0: "b", 1: "s"},
            "logits4": {0: "b"},
        },
        opset_version=17,
        dynamo=False,
    )
    secs["fp32_export_s"] = time.time() - t0
    slim = lm = base = None
    gc.collect()
    t1 = time.time()
    q = d / "guard-int8.onnx"
    try:
        quantize_dynamic(
            str(d / "guard-fp32.onnx"),
            str(q),
            weight_type=QuantType.QInt8,
            use_external_data_format=True,
        )
    except TypeError:  # newer ORT dropped the flag and picks external data by size
        quantize_dynamic(str(d / "guard-fp32.onnx"), str(q), weight_type=QuantType.QInt8)
    secs["int8_quantize_s"] = time.time() - t1
    return secs


def measure(which: str, base_model: str, res: dict[str, Any]) -> None:
    from breachlab.common.config import ServeConfig
    from breachlab.eval.latency import run_latency
    from breachlab.serve.onnx_backend import load_guard
    from breachlab.train.prompts import encode_prompt
    from breachlab.train.scoring import read_rows

    serve = ServeConfig()
    data_dir = Path(os.environ["DATA_DIR"])
    gold = read_rows(data_dir / "gold_test.jsonl")
    capped = [r.text[:500] for r in gold]  # the gateway caps messages at 500 characters
    d = OUT / f"export3-{which}"
    entry: dict[str, Any] = {"base_model": base_model}
    res[which] = entry
    save(res)
    try:
        entry["timings"] = export_onnx(base_model, find_adapter(which), serve.token_ids(), d)
    except Exception:  # noqa: BLE001 - recorded, run continues
        entry["export_error"] = traceback.format_exc()[-1500:]
        save(res)
        return
    entry["files_mb"] = {
        f.name: round(f.stat().st_size / 1e6) for f in d.iterdir() if f.stat().st_size > 1_000_000
    }
    save(res)
    threads = os.cpu_count() or 4
    for variant, fname in (("fp32", "guard-fp32.onnx"), ("dyn_int8", "guard-int8.onnx")):
        v: dict[str, Any] = {}
        entry[variant] = v
        try:
            m = load_guard(d, serve, which, threads=threads, file=fname)
            sample = gold[:300]
            lg = np.stack(
                [
                    m.logits_fn(encode_prompt(m.encode, r.text, m.words, "input", 512))
                    for r in sample
                ]
            )
            np.save(OUT / f"gold300_{which}_{variant}_logits.npy", lg)
            for rung, limit in (("tok512", 512), ("tok128", 128)):
                m.max_tokens = {"input_guard": limit, "output_filter": min(limit, 384)}
                stats = run_latency(m.score, capped, requests=REQUESTS, concurrency=1)
                v[f"{rung}_c1"] = stats.__dict__
                save(res)
            del m
        except Exception:  # noqa: BLE001
            v["error"] = traceback.format_exc()[-1500:]
        gc.collect()
        save(res)
    if which not in KEEP_ONNX:  # keep kernel output small; the JSON/NPY files are what we fetch
        shutil.rmtree(d, ignore_errors=True)


def main() -> None:
    res: dict[str, Any] = {"cpu_threads": os.cpu_count(), "requests": REQUESTS, "specs": SPECS}
    save(res)
    for which, base_model in SPECS:
        try:
            measure(which, base_model, res)
        except Exception:  # noqa: BLE001
            res.setdefault(which, {})["error"] = traceback.format_exc()[-1500:]
            save(res)
        gc.collect()
    print(json.dumps(res, default=str)[:4000])


if SPECS:
    main()
