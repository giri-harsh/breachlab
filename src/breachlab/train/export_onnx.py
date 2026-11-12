"""Stage 5: merge LoRA (or take the untuned base) -> slim ONNX (hidden state -> 4 verbalizer logits) -> variants.

The exported graph returns only the 4 verbalizer logits at the last position, so the 151k-way LM head is
never evaluated (PRD 7.2: one forward pass, 4-way softmax). Heavy deps are imported lazily; the export is a
CPU kernel job (not unit-tested). The pure helpers (calibration lookup, wrapper maths, file manifest) are.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Untuned base model: no calibration was fitted (zero-shot T hit the search bound, ADR-0032), so it serves with
# the identity temperature and the default 0.5 threshold, which is exactly the "weak Level-3 guard" behaviour.
BASE_CALIBRATION: dict[str, float] = {"temperature": 1.0, "threshold": 0.5}
VARIANTS = ("fp32", "kv", "w8", "dyn_int8", "dyn_pc")
VARIANT_FILES = {
    "fp32": "guard-fp32.onnx",
    "kv": "guard-kv.onnx",
    "w8": "guard-w8.onnx",
    "dyn_int8": "guard-int8.onnx",
    "dyn_pc": "guard-dynpc.onnx",
}


def score_from_hidden(
    hidden_last: NDArray[np.float64], head_rows: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Reference maths of the exported head: (B, H) @ (4, H)^T -> (B, 4)."""
    return np.asarray(hidden_last @ head_rows.T, dtype=np.float64)


def calibration_for(which: str, results: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Pick the calibration (temperature, threshold) of ``which`` (sft-*, dpo-*, base) from results.json dicts.

    A results.json holds ``{"sft": {"calibration": ...}, "dpo": {"calibration": ...}}`` for a run tag. The first
    file that carries the requested key wins; ``base`` has no fitted calibration (BASE_CALIBRATION).
    """
    if which == "base":
        return dict(BASE_CALIBRATION)
    key = "dpo" if which.startswith("dpo") else "sft"
    for res in results:
        cal = res.get(key, {}).get("calibration") if isinstance(res.get(key), dict) else None
        if cal:
            return {k: float(v) for k, v in cal.items()}
    raise LookupError(f"no calibration for {which!r} in {len(results)} results.json files")


def load_results(root: Path) -> list[dict[str, Any]]:
    """Every results.json under ``root`` that parses as a JSON object (tolerates unrelated files)."""
    out: list[dict[str, Any]] = []
    for f in sorted(root.rglob("results.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def manifest(out_dir: Path) -> dict[str, int]:
    """File name -> size in bytes for everything the export produced (goes into export.json)."""
    return {p.name: p.stat().st_size for p in sorted(out_dir.iterdir()) if p.is_file()}


def export_onnx(  # pragma: no cover - heavy
    base_model: str,
    adapter: Path | None,
    verbalizer_ids: list[int],
    out_dir: Path,
    variants: Sequence[str] = ("fp32", "w8"),
) -> dict[str, Path]:
    """Export one model; ``adapter=None`` exports the untuned base. Returns variant -> .onnx path."""
    import onnx
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(adapter) if adapter else base_model)
    lm = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
    if adapter is not None:
        from peft import PeftModel

        lm = PeftModel.from_pretrained(lm, str(adapter)).merge_and_unload()
    lm = lm.eval()

    class Slim(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.body = lm.model
            self.register_buffer("head", lm.lm_head.weight[verbalizer_ids].detach().clone())

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            pos = (attention_mask.cumsum(-1) - 1).clamp(min=0)
            h = self.body(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=pos
            ).last_hidden_state
            return h[:, -1, :] @ self.head.T

    slim = Slim().eval()
    ids = torch.tensor([tok.encode("<|im_start|>user\nhello<|im_end|>", add_special_tokens=False)])
    raw = out_dir / "_raw"
    raw.mkdir(exist_ok=True)
    torch.onnx.export(
        slim,
        (ids, torch.ones_like(ids)),
        str(raw / "guard.onnx"),
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
    # The exporter scatters one file per initializer once the graph exceeds 2 GB; pack them into one .data file.
    fp32 = out_dir / VARIANT_FILES["fp32"]
    model = onnx.load(str(raw / "guard.onnx"))
    onnx.save_model(
        model,
        str(fp32),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=fp32.name + ".data",
        size_threshold=1024,
    )
    shutil.rmtree(raw, ignore_errors=True)
    made: dict[str, Path] = {}
    if "fp32" in variants:
        made["fp32"] = fp32
    if "kv" in variants:
        try:
            from breachlab.common.config import ServeConfig
            from breachlab.train.export_kv import KV_FILE, export_kv

            report = export_kv(lm, tok, verbalizer_ids, ServeConfig().words(), out_dir, fp32)
            made["kv"] = out_dir / KV_FILE
            (out_dir / "kv_report.json").write_text(json.dumps(report), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - optional variant; the error text is kept for the ladder report
            import traceback

            (out_dir / "kv_error.txt").write_text(traceback.format_exc()[-3000:], encoding="utf-8")
            print(f"kv export failed: {type(exc).__name__}: {exc}")
    if "dyn_int8" in variants:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = out_dir / VARIANT_FILES["dyn_int8"]
        quantize_dynamic(str(fp32), str(q), weight_type=QuantType.QInt8)
        made["dyn_int8"] = q
    if "dyn_pc" in variants:
        # Dynamic int8 with per-channel weights, keeping the MLP down projections (largest activation outliers)
        # in fp32: the plain per-tensor variant lost AUROC 0.994 -> 0.979 and blew FPR up (ADR-0035).
        from onnxruntime.quantization import QuantType, quantize_dynamic

        skip = [
            n.name
            for n in onnx.load(str(fp32), load_external_data=False).graph.node
            if "down_proj" in n.name
        ]
        pc = out_dir / VARIANT_FILES["dyn_pc"]
        quantize_dynamic(
            str(fp32),
            str(pc),
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["MatMul"],
            nodes_to_exclude=skip,
        )
        made["dyn_pc"] = pc
    if "w8" in variants:
        # Weight-only int8 (MatMulNBits, 8 bit): dynamic int8 also quantizes activations, which hurt quality (ADR-0035).
        try:
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

            w = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=8, block_size=32, is_symmetric=True)
            w.process()
            target = out_dir / VARIANT_FILES["w8"]
            w.model.save_model_to_file(str(target), use_external_data_format=True)
            made["w8"] = target
        except Exception as exc:  # noqa: BLE001 - optional variant; the report records the failure
            (out_dir / "w8_error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
    tok.save_pretrained(out_dir)
    return made


def export_int8(  # pragma: no cover - heavy
    base_model: str, adapter: Path, verbalizer_ids: list[int], out_dir: Path
) -> Path:
    """Backwards-compatible entry point: fp32 + dynamic int8 + weight-only int8 of one adapter."""
    made = export_onnx(base_model, adapter, verbalizer_ids, out_dir, ("fp32", "dyn_int8", "w8"))
    return made["dyn_int8"]
