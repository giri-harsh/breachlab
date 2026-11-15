"""Distil the v2 0.6B guard (teacher) into a small multilingual encoder student (PRD 7, latency; ADR-0105).

Runs inside a Kaggle GPU kernel (``distill_job.py``): scores the training rows with the teacher adapter (4 verbalizer
logits per row = soft labels), trains each candidate student on soft plus hard labels, calibrates on val, exports a
dynamic-int8 ONNX graph, checks parity and writes the logits the offline cascade analysis needs. The pure parts
(head+tail truncation, student calibration helpers) are unit tested; the GPU parts are marked ``no cover``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from breachlab.data.schema import Row
from breachlab.train.prompts import label_index, wrap_ids

Arr = NDArray[np.float64]


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str  # HF id; licence checked in ADR-0105 (MIT or Apache-2.0 only)
    tokenizer: str
    lr: float
    licence: str


CANDIDATES: dict[str, Candidate] = {
    "minilm-ml": Candidate(
        "minilm-ml",
        "microsoft/Multilingual-MiniLM-L12-H384",
        "FacebookAI/xlm-roberta-base",
        1e-4,
        "MIT",
    ),
    "distilbert-ml": Candidate(
        "distilbert-ml",
        "distilbert/distilbert-base-multilingual-cased",
        "distilbert/distilbert-base-multilingual-cased",
        6e-5,
        "Apache-2.0",
    ),
    "xlmr-base": Candidate(
        "xlmr-base", "FacebookAI/xlm-roberta-base", "FacebookAI/xlm-roberta-base", 3e-5, "MIT"
    ),
}


def pad_batch(seqs: list[list[int]], pad_id: int) -> tuple[list[list[int]], list[list[int]]]:
    n = max(len(s) for s in seqs)
    ids = [s + [pad_id] * (n - len(s)) for s in seqs]
    mask = [[1] * len(s) + [0] * (n - len(s)) for s in seqs]
    return ids, mask


def hard_labels(rows: list[Row]) -> NDArray[np.int64]:
    return np.array([label_index(r) for r in rows], dtype=np.int64)


def agreement(teacher: Arr, student: Arr) -> dict[str, float]:
    """Decision agreement (argmax over the 4 classes, and SAFE vs UNSAFE) between teacher and student logits."""
    t4, s4 = teacher.argmax(-1), student.argmax(-1)
    return {
        "argmax4": float((t4 == s4).mean()),
        "binary": float(((t4 == 0) == (s4 == 0)).mean()),
    }


def read_rows(path: Path) -> list[Row]:
    return [
        Row.model_validate(json.loads(x)) for x in path.read_text("utf-8").split("\n") if x.strip()
    ]


def teacher_logits(  # pragma: no cover - GPU
    adapter_dir: Path, base_model: str, splits: dict[str, list[Row]], out: Path
) -> dict[str, Arr]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from breachlab.common.config import ServeConfig
    from breachlab.train.score_model import score_rows

    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16).cuda()
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    serve = ServeConfig()
    res: dict[str, Arr] = {}
    for name, rows in splits.items():
        t0 = time.time()
        res[name] = score_rows(model, tok, rows, serve, 512, 32)
        np.save(out / f"teacher_{name}.npy", res[name])
        print(f"teacher {name}: {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)
    del model, base
    torch.cuda.empty_cache()
    return res


def train_student(  # pragma: no cover - GPU
    cand: Candidate,
    splits: dict[str, list[Row]],
    teacher: dict[str, Arr],
    out: Path,
    max_len: int = 128,
    epochs: int = 4,
    batch: int = 32,
    train_names: tuple[str, ...] = ("train",),
    kd_temp: float = 2.0,
    alpha: float = 0.7,
    attack_weight: float = 1.0,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(0)
    np.random.seed(0)
    tok = AutoTokenizer.from_pretrained(cand.tokenizer)
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def enc(rows: list[Row]) -> list[list[int]]:
        ids = tok([r.text for r in rows], add_special_tokens=False)["input_ids"]
        return [wrap_ids(list(i), max_len, cls_id, sep_id) for i in ids]

    data = {k: enc(v) for k, v in splits.items()}
    model = AutoModelForSequenceClassification.from_pretrained(cand.model, num_labels=4).cuda()
    train_ids = [ids for n in train_names for ids in data[n]]
    tl = torch.tensor(np.concatenate([teacher[n] for n in train_names]), dtype=torch.float32)
    yh = torch.tensor(np.concatenate([hard_labels(splits[n]) for n in train_names]))
    cw = torch.tensor([1.0, attack_weight, attack_weight, attack_weight]).cuda()
    n = len(train_ids)
    steps = epochs * ((n + batch - 1) // batch)
    opt = torch.optim.AdamW(model.parameters(), lr=cand.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / (0.06 * steps), max(0.0, (steps - s) / (0.94 * steps)))
    )
    scaler = torch.amp.GradScaler()
    T = kd_temp

    def forward(idx: list[int] | None, seqs: list[list[int]]) -> Any:
        ids, mask = pad_batch(seqs, pad_id)
        with torch.autocast("cuda", dtype=torch.float16):
            return model(
                input_ids=torch.tensor(ids).cuda(), attention_mask=torch.tensor(mask).cuda()
            ).logits.float()

    def predict(seqs: list[list[int]], bs: int = 128) -> Arr:
        model.eval()
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        out_arr = np.zeros((len(seqs), 4))
        with torch.no_grad():
            for s in range(0, len(order), bs):
                chunk = order[s : s + bs]
                out_arr[chunk] = forward(None, [seqs[i] for i in chunk]).cpu().numpy()
        model.train()
        return out_arr

    def val_auc() -> float:
        lg = predict(data["val"])
        p = 1 - np.exp(lg - lg.max(-1, keepdims=True))[:, 0] / np.exp(
            lg - lg.max(-1, keepdims=True)
        ).sum(-1)
        y = np.array([r.label == "UNSAFE" for r in splits["val"]], dtype=int)
        return float(roc_auc_score(y, p))

    best: tuple[float, dict[str, Any]] = (-1.0, {})
    hist: list[dict[str, float]] = []
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, batch):
            b = perm[s : s + batch].tolist()
            logits = forward(b, [train_ids[i] for i in b])
            kd = F.kl_div(
                F.log_softmax(logits / T, -1),
                F.softmax(tl[b].cuda() / T, -1),
                reduction="batchmean",
            ) * (T * T)
            ce = F.cross_entropy(logits, yh[b].cuda(), weight=cw)
            loss = alpha * kd + (1 - alpha) * ce
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        auc = val_auc()
        hist.append({"epoch": float(ep + 1), "val_auroc": auc, "seconds": time.time() - t0})
        print(f"{cand.name} epoch {ep + 1}: val AUROC {auc:.5f}", flush=True)
        if auc > best[0]:
            best = (auc, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    dest = out / cand.name
    dest.mkdir(parents=True, exist_ok=True)
    held = {k: v for k, v in data.items() if k not in train_names}
    logits_out = {k: predict(v) for k, v in held.items()}
    for k, v in logits_out.items():
        np.save(dest / f"logits_{k}_torch.npy", v)
    tok.backend_tokenizer.save(str(dest / "tokenizer.json"))
    meta = {
        "candidate": cand.name,
        "model": cand.model,
        "licence": cand.licence,
        "cls_id": cls_id,
        "sep_id": sep_id,
        "pad_id": pad_id,
        "max_len": max_len,
        "epochs": epochs,
        "train_names": list(train_names),
        "kd_temp": kd_temp,
        "alpha": alpha,
        "attack_weight": attack_weight,
        "history": hist,
        "train_seconds": time.time() - t0,
        "agreement_vs_teacher": {
            k: agreement(teacher[k], logits_out[k]) for k in logits_out if k in teacher
        },
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    export_int8(model, dest, pad_id, held, logits_out, meta)
    return meta


def export_int8(  # pragma: no cover - heavy
    model: Any,
    dest: Path,
    pad_id: int,
    data: dict[str, list[list[int]]],
    torch_logits: dict[str, Arr],
    meta: dict[str, Any],
) -> None:
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model.cpu().eval()
    dummy = torch.tensor([[meta["cls_id"], 5, 6, meta["sep_id"]]])
    fp32 = dest / "student-fp32.onnx"
    torch.onnx.export(
        model,
        (dummy, torch.ones_like(dummy)),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}},
        opset_version=17,
        dynamo=False,
    )
    int8 = dest / "student-int8.onnx"
    quantize_dynamic(
        str(fp32),
        str(int8),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Gather"],
    )
    meta["bytes"] = {"fp32": fp32.stat().st_size, "int8": int8.stat().st_size}
    sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    parity: dict[str, Any] = {}
    for split in data:
        seqs, outs = data[split], np.zeros((len(data[split]), 4))
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        t0 = time.time()
        for s in range(0, len(order), 64):
            chunk = order[s : s + 64]
            ids, mask = pad_batch([seqs[i] for i in chunk], pad_id)
            outs[chunk] = sess.run(
                ["logits"],
                {
                    "input_ids": np.array(ids, dtype=np.int64),
                    "attention_mask": np.array(mask, dtype=np.int64),
                },
            )[0]
        np.save(dest / f"logits_{split}_int8.npy", outs)
        parity[split] = {
            **agreement(torch_logits[split], outs),
            "max_abs_logit_diff": float(np.abs(torch_logits[split] - outs).max()),
            "seconds_batched": time.time() - t0,
        }
    meta["int8_parity_vs_torch"] = parity
    (dest / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
