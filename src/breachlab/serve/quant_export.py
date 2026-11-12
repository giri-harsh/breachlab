"""Single-file weight-only int4 / int8 guard for a small CPU host (ADR-0080, lane B3).

The champion prefix-KV graph (``guard-kv.onnx``, fp32, 2.4 GB incl. a 622 MB token embedding) is rewritten into one
self-contained ``.onnx`` file (no external data) so a serverless function can pull it into RAM:

* every ``MatMul`` with a constant weight becomes ``MatMulNBits`` (4 or 8 bit, symmetric, block 32 or 64); the
  4-row verbalizer head stays fp32;
* the token embedding (``Gather``, 151,936 x 1,024) becomes a row-wise int8 table + per-row scale (``int8`` mode), a
  4-bit ``GatherBlockQuantized`` (``int4`` mode) or stays fp32 (``fp32`` mode, diagnostic).

Quality is judged against the fp32 graph on all gold rows at the shipping calibration (``judge``): the AUROC may drop
at most 0.002 and the FPR / ASR of the variant must lie inside the bootstrap 95% interval of the fp32 graph. The
ADR-0057 rule (``eval.parity``) is reported next to it. Everything that touches weights runs in a Kaggle CPU kernel
(``main``); the pure helpers (spec table, bootstrap, verdict) are unit-tested without any model.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

AUROC_MAX_DROP = 0.002  # lane B3 rule (tighter than ADR-0057's 0.003)
CLIP_CHARS = 500
EMBED_MIN_ROWS = (
    100_000  # the token embedding is the only constant Gather with a vocabulary-sized table
)
HEAD_MAX_DIM = 64  # a constant MatMul weight with a side below this is the 4-class verbalizer head


@dataclass(frozen=True)
class VariantSpec:
    name: str
    bits: int  # MatMulNBits width
    block: int
    embed: str  # fp32 | int8 | int4
    accuracy_level: int | None = (
        None  # MatMulNBits compute type (4 = int8 activations); None = fp32 compute
    )
    ship: bool = False  # keep the file in the kernel output (the diagnostics are deleted)
    sym: bool = True  # symmetric (no zero points) vs asymmetric block quantization
    algo: str = "rtn"  # rtn (round to nearest) | hqq (half-quadratic, data free)
    down_bits: int | None = None  # width of the MLP down projections when it differs from ``bits``


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("q4b32", 4, 32, "int8", ship=True),
    VariantSpec("q4b32_asym", 4, 32, "int8", sym=False, ship=True),
    VariantSpec("q4b16_asym", 4, 16, "int8", sym=False, ship=True),
    VariantSpec("q4m8d", 4, 32, "int8", down_bits=8, ship=True),
    VariantSpec("q4m8d_asym", 4, 32, "int8", sym=False, down_bits=8, ship=True),
    VariantSpec("q4b32_hqq", 4, 32, "int8", algo="hqq", ship=True),
    VariantSpec("q8b32", 8, 32, "int8", ship=True),
    VariantSpec("q8b32_acc4", 8, 32, "int8", accuracy_level=4, ship=True),
    VariantSpec("q4b64", 4, 64, "int8", ship=True),
    VariantSpec("q4b32_f32e", 4, 32, "fp32"),
    VariantSpec(
        "q4b32_g4", 4, 32, "int4"
    ),  # known to fail: forcing opset 21 breaks ReduceMean (smoke run)
)
BY_NAME = {v.name: v for v in VARIANTS}


def select(names: Sequence[str]) -> list[VariantSpec]:
    """Variants by name, in the order given; unknown names are an error (typos must not silently skip a run)."""
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        raise KeyError(f"unknown variants {unknown}; known: {sorted(BY_NAME)}")
    return [BY_NAME[n] for n in names]


# ----------------------------------------------------------------------------------------------- pure statistics


def rate_ci(
    flags: NDArray[np.bool_], n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean of a boolean array and its bootstrap percentile interval (rate, lo, hi)."""
    n = len(flags)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = flags.astype(np.float64)[idx].mean(axis=1)
    a = (1.0 - level) / 2.0 * 100.0
    return (
        float(flags.mean()),
        float(np.percentile(means, a)),
        float(np.percentile(means, 100.0 - a)),
    )


def judge(
    auroc_ref: float,
    auroc_var: float,
    fpr_var: float,
    asr_var: float,
    fpr_ci: tuple[float, float, float],
    asr_ci: tuple[float, float, float],
    adr0057_ok: bool,
) -> dict[str, Any]:
    """The lane B3 acceptance rule plus the ADR-0057 verdict, with the individual checks spelled out."""
    drop = auroc_ref - auroc_var
    checks = {
        "auroc_drop_ok": bool(drop <= AUROC_MAX_DROP + 1e-12),
        "fpr_in_fp32_ci": bool(fpr_ci[1] <= fpr_var <= fpr_ci[2]),
        "asr_in_fp32_ci": bool(asr_ci[1] <= asr_var <= asr_ci[2]),
    }
    return {
        "auroc_drop": float(drop),
        **checks,
        "adr0057_ok": bool(adr0057_ok),
        "pass": bool(all(checks.values())),
    }


# --------------------------------------------------------------------------------------------- graph rewriting


def _consumers(graph: Any, name: str) -> int:
    return sum(1 for n in graph.node for i in n.input if i == name)


def find_embedding(graph: Any) -> Any:
    """The ``Gather`` over the vocabulary-sized constant table (raises if there is not exactly one)."""
    inits = {i.name: i for i in graph.initializer}
    hits = [
        n
        for n in graph.node
        if n.op_type == "Gather"
        and n.input[0] in inits
        and len(inits[n.input[0]].dims) == 2
        and inits[n.input[0]].dims[0] >= EMBED_MIN_ROWS
    ]
    if len(hits) != 1:
        raise LookupError(f"expected one embedding Gather, found {len(hits)}")
    return hits[0]


def head_nodes(graph: Any) -> list[str]:
    """Names of constant-weight MatMuls that must stay fp32 (the 4-row verbalizer head)."""
    inits = {i.name: i for i in graph.initializer}
    out: list[str] = []
    for n in graph.node:
        if n.op_type != "MatMul":
            continue
        w = [inits[i] for i in n.input if i in inits]
        if w and min(w[0].dims) < HEAD_MAX_DIM:
            out.append(n.name)
    return out


def embedding_to_int8(
    model: Any,
) -> dict[str, float]:  # pragma: no cover - needs onnx + the weights
    """Replace the embedding Gather by int8 rows + per-row fp32 scales (Gather, Cast, Gather, Mul)."""
    from onnx import TensorProto, helper, numpy_helper

    g = model.graph
    node = find_embedding(g)
    init = next(i for i in g.initializer if i.name == node.input[0])
    w = numpy_helper.to_array(init).astype(np.float32)
    scale = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-8) / 127.0
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    err = float(np.abs(q.astype(np.float32) * scale - w).max())
    base = node.name or "embed"
    qn, sn = base + "_q8", base + "_rowscale"
    g.initializer.extend(
        [numpy_helper.from_array(q, qn), numpy_helper.from_array(scale.astype(np.float32), sn)]
    )
    ids, out = node.input[1], node.output[0]
    new = [
        helper.make_node("Gather", [qn, ids], [base + "_gq"], name=base + "_gq", axis=0),
        helper.make_node(
            "Cast", [base + "_gq"], [base + "_gf"], name=base + "_cast", to=TensorProto.FLOAT
        ),
        helper.make_node("Gather", [sn, ids], [base + "_gs"], name=base + "_gs", axis=0),
        helper.make_node("Mul", [base + "_gf", base + "_gs"], [out], name=base + "_mul"),
    ]
    nodes = []
    for n in g.node:
        nodes.extend(new if n is node or n.name == node.name else [n])
    g.ClearField("node")
    g.node.extend(nodes)
    if _consumers(g, init.name) == 0:
        g.initializer.remove(init)
    return {"max_abs_weight_error": err, "rows": float(q.shape[0]), "dim": float(q.shape[1])}


def build_variant(
    src: Path, spec: VariantSpec
) -> tuple[Any, dict[str, Any]]:  # pragma: no cover - heavy
    """Load the fp32 prefix-KV graph and return (quantized ModelProto, info)."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    model = onnx.load(str(src))  # external data is read into the proto
    info: dict[str, Any] = {"spec": asdict(spec)}
    exclude = head_nodes(model.graph)
    info["fp32_matmuls_kept"] = exclude
    if spec.embed == "int8":
        info["embedding_int8"] = embedding_to_int8(model)
        ops: tuple[str, ...] = ("MatMul",)
    elif spec.embed == "int4":
        emb = find_embedding(model.graph)
        exclude = exclude + [
            n.name for n in model.graph.node if n.op_type == "Gather" and n.name != emb.name
        ]
        ops = ("MatMul", "Gather")
    else:
        ops = ("MatMul",)
    down = [n.name for n in model.graph.node if n.op_type == "MatMul" and "down_proj" in n.name]
    mixed = spec.down_bits is not None
    t0 = time.perf_counter()
    if spec.algo == "hqq":
        from onnxruntime.quantization.matmul_nbits_quantizer import HQQWeightOnlyQuantConfig

        q = MatMulNBitsQuantizer(
            model,
            nodes_to_exclude=exclude,
            algo_config=HQQWeightOnlyQuantConfig(block_size=spec.block, bits=spec.bits),
        )
    else:
        q = MatMulNBitsQuantizer(
            model,
            bits=spec.bits,
            block_size=spec.block,
            is_symmetric=spec.sym,
            accuracy_level=spec.accuracy_level,
            nodes_to_exclude=exclude + (down if mixed else []),
            op_types_to_quantize=ops,
        )
    q.process()
    if mixed:  # second pass: the down projections (largest activation outliers) at a wider width
        info["down_proj_matmuls"] = len(down)
        q = MatMulNBitsQuantizer(
            q.model.model,
            bits=int(spec.down_bits or 8),
            block_size=spec.block,
            is_symmetric=spec.sym,
            accuracy_level=spec.accuracy_level,
            nodes_to_exclude=exclude,
            op_types_to_quantize=("MatMul",),
        )
        q.process()
    info["quantize_s"] = time.perf_counter() - t0
    return q.model, info


def save_single_file(onnx_model: Any, target: Path) -> int:  # pragma: no cover - heavy
    """Write one self-contained .onnx (no external data); returns its size in bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx_model.save_model_to_file(str(target), use_external_data_format=False)
    return target.stat().st_size


# ----------------------------------------------------------------------------------------------- measurements


def rss_mb() -> float:
    """Peak resident set size (VmHWM) of this process in MB.

    ``ru_maxrss`` is not usable here: a child started by fork+exec inherits the parent's high-water mark
    (the smoke run reported 4.6-5.8 GB for every variant), while ``/proc/self/status`` is per address space."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def pct(values: Sequence[float], q: float) -> float:
    return (
        float(np.percentile(np.asarray(values, dtype=np.float64), q))
        if len(values)
        else float("nan")
    )


def probe(  # pragma: no cover - needs onnxruntime + weights
    model_dir: Path, file: str, mode: str, threads: int, gold: Path, rows: int
) -> dict[str, Any]:
    """Cold-start footprint and latency of one variant as the serverless function runs it: file read into RAM
    (``bytes``) or opened by path, one ORT session, ``threads`` threads. Meant to run in its own process."""
    from breachlab.common.config import ServeConfig
    from breachlab.serve.onnx_backend import load_guard
    from breachlab.serve.trunc_eval import read_gold
    from breachlab.train.prompts import encode_prompt

    t0 = time.perf_counter()
    blob = (model_dir / file).read_bytes() if mode == "bytes" else None
    read_s = time.perf_counter() - t0
    rss_read = rss_mb()
    t1 = time.perf_counter()
    model = load_guard(
        model_dir, ServeConfig(), "b3", threads=threads, file=file, model_bytes=blob, kv=True
    )
    load_s = time.perf_counter() - t1
    del blob
    rss_load = rss_mb()
    texts = [r.text[:CLIP_CHARS] for r in read_gold(gold)[:rows]]
    lat: list[float] = []
    for i, text in enumerate(texts):
        ids = encode_prompt(model.encode, text, model.words, "input", 512)
        s = time.perf_counter()
        model.logits_fn(ids)
        if i >= 3:  # three warm-up requests, then the measured ones
            lat.append((time.perf_counter() - s) * 1000.0)
    return {
        "file": file,
        "mode": mode,
        "threads": threads,
        "read_s": read_s,
        "ort_load_s": load_s,
        "rss_peak_after_read_mb": rss_read,
        "rss_peak_after_load_mb": rss_load,
        "rss_peak_end_mb": rss_mb(),
        "n_requests": len(lat),
        "model_p50_ms": pct(lat, 50),
        "model_p95_ms": pct(lat, 95),
        "model_mean_ms": float(np.mean(lat)) if lat else float("nan"),
    }


def score_gold(  # pragma: no cover - needs onnxruntime + weights
    model_dir: Path, file: str, gold: Path, threads: int, rows: int
) -> NDArray[np.float64]:
    from breachlab.common.config import ServeConfig
    from breachlab.serve.onnx_backend import load_guard
    from breachlab.serve.trunc_eval import read_gold
    from breachlab.train.prompts import encode_prompt

    model = load_guard(model_dir, ServeConfig(), "b3", threads=threads, file=file, kv=True)
    out = []
    for i, r in enumerate(read_gold(gold)[:rows]):
        ids = encode_prompt(model.encode, r.text[:CLIP_CHARS], model.words, "input", 512)
        out.append(np.asarray(model.logits_fn(ids), dtype=np.float64))
        if i % 250 == 0:
            print(f"  scored {i} rows", flush=True)
    return np.stack(out)


def compare(
    labels: Sequence[int], ref: NDArray[np.float64], var: NDArray[np.float64], t: float, thr: float
) -> dict[str, Any]:
    """Parity of one variant against the fp32 logits at the shipping calibration."""
    from breachlab.eval.parity import p_unsafe, variant_parity

    par = variant_parity(list(labels), ref, var, t, thr)
    y = np.asarray(labels, dtype=np.int64)
    pr = p_unsafe(ref, t)
    fpr_ci = rate_ci(pr[y == 0] >= thr)
    asr_ci = rate_ci(pr[y == 1] < thr)
    verdict = judge(
        par.auroc_torch, par.auroc_variant, par.fpr_variant, par.asr_variant, fpr_ci, asr_ci, par.ok
    )
    cat_agree = float(np.mean(np.argmax(ref[:, 1:], axis=1) == np.argmax(var[:, 1:], axis=1)))
    return {
        "parity": par.as_dict(),
        "fp32_fpr_ci95": list(fpr_ci),
        "fp32_asr_ci95": list(asr_ci),
        "category_argmax_agreement": cat_agree,
        "verdict": verdict,
    }


# ------------------------------------------------------------------------------------------------- kernel entry


def find_source(
    root: Path, dir_name: str = "export-sft-0.6b-s0"
) -> Path:  # pragma: no cover - kernel
    found = sorted(p for p in root.rglob(dir_name) if (p / "guard-kv.onnx").exists())
    if not found:
        raise FileNotFoundError(f"no {dir_name}/guard-kv.onnx under {root}")
    return found[0]


def stage_dir(src: Path, dest: Path) -> None:  # pragma: no cover - kernel
    """Small companions of the ONNX file (tokenizer, calibration, prefix K/V) next to the variant."""
    import shutil

    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "tokenizer.json",
        "calibration.json",
        "prefix_kv_input.npz",
        "prefix_kv_output.npz",
    ):
        shutil.copy2(src / name, dest / name)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - kernel entry
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/b3"))
    ap.add_argument(
        "--dir-name",
        default="export-sft-0.6b-s0",
        help="export folder name in the source kernel (base: export-base)",
    )
    ap.add_argument("--variants", default=",".join(v.name for v in VARIANTS))
    ap.add_argument("--rows", type=int, default=-1, help="gold rows to score (-1 all)")
    ap.add_argument("--probe-rows", type=int, default=43)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument(
        "--no-reference",
        action="store_true",
        help="score the variants only (no fp32 reference, no probes, no parity): logits for recalibration",
    )
    ap.add_argument(
        "--probe", nargs=4, metavar=("DIR", "FILE", "MODE", "THREADS"), help="internal: one probe"
    )
    a = ap.parse_args(argv)

    if a.probe:  # a fresh process, so ru_maxrss belongs to this variant only
        d, f, mode, th = a.probe
        print(
            "PROBE " + json.dumps(probe(Path(d), f, mode, int(th), a.gold, a.probe_rows)),
            flush=True,
        )
        return 0

    from breachlab.serve.trunc_eval import read_gold

    src = find_source(a.input_root, a.dir_name)
    cal = json.loads((src / "calibration.json").read_text("utf-8"))
    t, thr = float(cal["temperature"]), float(cal["threshold"])
    labels = [int(r.label == "UNSAFE") for r in read_gold(a.gold)]
    n = len(labels) if a.rows < 0 else a.rows
    labels = labels[:n]
    a.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "source": str(src),
        "temperature": t,
        "threshold": thr,
        "n_rows": n,
        "n_attacks": int(sum(labels)),
        "auroc_max_drop": AUROC_MAX_DROP,
        "variants": {},
    }

    def flush() -> None:
        (a.out / "b3_quant.json").write_text(json.dumps(report, indent=2), "utf-8")

    # 0. fp32 reference with the same graph and code path (prefix-KV fp32 == plain fp32 up to 1.5e-5).
    ref: NDArray[np.float64] | None = None
    if not a.no_reference:
        print("scoring the fp32 reference", flush=True)
        t0 = time.perf_counter()
        ref = score_gold(src, "guard-kv.onnx", a.gold, a.threads, n)
        report["reference"] = {
            "wall_s": time.perf_counter() - t0,
            "bytes": (src / "guard-kv.onnx.data").stat().st_size,
        }
        np.save(a.out / "logits_fp32.npy", ref)
        report["reference"]["probe_1thread"] = _run_probe(src, "guard-kv.onnx", "path", a)
        flush()

    for spec in select([x for x in a.variants.split(",") if x]):
        print(f"=== {spec.name}", flush=True)
        vd = a.out / spec.name
        entry: dict[str, Any] = {"spec": asdict(spec)}
        report["variants"][spec.name] = entry
        try:
            model, info = build_variant(src / "guard-kv.onnx", spec)
            entry.update(info)
            fname = f"guard-{spec.name}.onnx"  # distinct from the fp32 guard-kv.onnx already in the Hub repo
            stage_dir(src, vd)
            entry["bytes"] = save_single_file(model, vd / fname)
            del model
            print(f"  wrote {spec.name}: {entry['bytes']} bytes", flush=True)
            if ref is not None:
                for mode in ("bytes", "path"):
                    entry[f"probe_{mode}_1thread"] = _run_probe(vd, fname, mode, a)
            flush()
            t0 = time.perf_counter()
            lg = score_gold(vd, fname, a.gold, a.threads, n)
            entry["score_wall_s"] = time.perf_counter() - t0
            np.save(a.out / f"logits_{spec.name}.npy", lg)
            if ref is not None:
                entry["max_abs_logit_diff"] = float(np.max(np.abs(lg - ref)))
                entry.update(compare(labels, ref, lg, t, thr))
                print(f"  verdict {spec.name}: {json.dumps(entry['verdict'])}", flush=True)
        except Exception as exc:  # noqa: BLE001 - a failed variant is a result, recorded with its error
            import traceback

            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback_tail"] = traceback.format_exc()[-2500:]
            print(f"  FAILED {spec.name}: {entry['error']}", flush=True)
        if not spec.ship:
            for f in vd.glob("*.onnx*"):
                f.unlink()
        flush()
    flush()
    return 0


def _run_probe(
    d: Path, file: str, mode: str, a: Any
) -> dict[str, Any]:  # pragma: no cover - kernel
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "breachlab.serve.quant_export",
        "--gold",
        str(a.gold),
        "--probe",
        str(d),
        file,
        mode,
        "1",
    ]
    cmd += ["--probe-rows", str(a.probe_rows)]
    pinned = ["taskset", "-c", "0", *cmd]  # one core, like a 1-vCPU function
    out = subprocess.run(pinned, capture_output=True, text=True, check=False)
    for line in out.stdout.splitlines():
        if line.startswith("PROBE "):
            return dict(json.loads(line[6:]))
    return {"error": (out.stdout + out.stderr)[-1500:], "returncode": out.returncode}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
