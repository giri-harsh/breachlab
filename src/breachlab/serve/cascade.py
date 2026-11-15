"""Student-then-teacher cascade guard (PRD 7 latency; ADR-0105).

A small multilingual encoder (the *student*, distilled from the 0.6B guard) scores every input message first. A
confident SAFE (``p_unsafe < t_low``) or confident UNSAFE (``p_unsafe > t_high``) answer is returned at once; the
uncertain band is escalated to the 0.6B guard (the *teacher*), which is loaded on the first escalation. If the teacher
cannot answer the uncertain band fails closed (UNSAFE). Output-filter calls (``kind="output"``) always use the
teacher: the student only saw user messages.

``choose_thresholds`` picks (t_low, t_high) on validation data so that the cascade's quality matches the teacher's
while keeping the escalation rate under a cap; it never looks at gold_test or the opener set.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from breachlab.serve.guard_runtime import GuardModel, GuardParams, GuardResult
from breachlab.train.prompts import LABELS, Kind
from breachlab.train.scoring import Verdict, probs_from_logits

Arr = NDArray[np.float64]
StudentLogits = Callable[[str], Arr]


@dataclass(frozen=True)
class CascadeParams:
    version: str
    temperature: float  # student calibration (val)
    t_low: float
    t_high: float


def student_verdict(logits: Arr, temperature: float, label: str) -> Verdict:
    p = probs_from_logits(logits, temperature)
    p_unsafe = float(1.0 - p[0])
    cat = LABELS[1 + int(np.argmax(p[1:]))] if label == "UNSAFE" else "none"
    return Verdict(label, cat, p_unsafe, (p[0], p[1], p[2], p[3]))


class CascadeModel(GuardModel):
    """Same ``score`` contract as ``GuardModel`` (the guard service is unchanged); adds the stage counters."""

    def __init__(
        self,
        student: StudentLogits,
        teacher: Callable[[], GuardModel],
        params: CascadeParams,
        max_tokens: dict[str, int],
    ) -> None:
        super().__init__(
            lambda s: [], lambda ids: np.zeros(4), [], GuardParams(params.version), max_tokens
        )
        self.student, self._teacher_loader, self.cparams = student, teacher, params
        self._teacher: GuardModel | None = None
        self.counts = {"student_safe": 0, "student_unsafe": 0, "teacher": 0, "failed_closed": 0}

    def teacher(self) -> GuardModel:
        if self._teacher is None:
            self._teacher = self._teacher_loader()
        return self._teacher

    def score(self, text: str, kind: Kind = "input") -> GuardResult:
        t0 = time.perf_counter()
        if kind == "input":
            logits = self.student(text)
            p = float(1.0 - probs_from_logits(logits, self.cparams.temperature)[0])
            if p < self.cparams.t_low:
                self.counts["student_safe"] += 1
                v = student_verdict(logits, self.cparams.temperature, "SAFE")
                return GuardResult(
                    v, self.cparams.version, (time.perf_counter() - t0) * 1000, "student"
                )
            if p > self.cparams.t_high:
                self.counts["student_unsafe"] += 1
                v = student_verdict(logits, self.cparams.temperature, "UNSAFE")
                return GuardResult(
                    v, self.cparams.version, (time.perf_counter() - t0) * 1000, "student"
                )
            try:
                r = self.teacher().score(text, kind)
            except Exception:  # noqa: BLE001 - the uncertain band fails closed when the teacher is unavailable
                self.counts["failed_closed"] += 1
                v = student_verdict(logits, self.cparams.temperature, "UNSAFE")
                return GuardResult(
                    v, self.cparams.version, (time.perf_counter() - t0) * 1000, "failed_closed"
                )
        else:
            r = self.teacher().score(text, kind)
        self.counts["teacher"] += 1
        return GuardResult(
            r.verdict, self.cparams.version, (time.perf_counter() - t0) * 1000, "teacher"
        )

    def escalation_rate(self) -> float:
        n = sum(self.counts.values())
        return (self.counts["teacher"] + self.counts["failed_closed"]) / n if n else 0.0


class StudentScorer:  # pragma: no cover - needs onnxruntime + weights
    """The int8 ONNX student: ``__call__(text)`` -> (4,) logits."""

    def __init__(
        self, onnx_path: str, tokenizer_path: str, meta: dict[str, Any], threads: int = 1
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        from breachlab.train.prompts import wrap_ids

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.meta, self._wrap = meta, wrap_ids

    def __call__(self, text: str) -> Arr:
        ids = list(self.tok.encode(text, add_special_tokens=False).ids)
        seq = self._wrap(
            ids, int(self.meta["max_len"]), int(self.meta["cls_id"]), int(self.meta["sep_id"])
        )
        arr = np.asarray([seq], dtype=np.int64)
        out = self.sess.run(["logits"], {"input_ids": arr, "attention_mask": np.ones_like(arr)})[0]
        return np.asarray(out[0], dtype=np.float64)


@dataclass(frozen=True)
class Thresholds:
    t_low: float
    t_high: float
    escalation: float
    asr: float
    fpr: float


def cascade_metrics(
    p_s: Arr, p_t: Arr, y: NDArray[np.int64], teacher_thr: float, t_low: float, t_high: float
) -> Thresholds:
    """Cascade ASR/FPR/escalation with student p_unsafe ``p_s`` and teacher p_unsafe ``p_t`` (teacher decides at ``teacher_thr``)."""
    esc = (p_s >= t_low) & (p_s <= t_high)
    pred = np.where(p_s < t_low, 0, np.where(p_s > t_high, 1, (p_t >= teacher_thr).astype(int)))
    atk, ben = y == 1, y == 0
    return Thresholds(
        t_low,
        t_high,
        float(esc.mean()),
        float((pred[atk] == 0).mean()) if atk.any() else 0.0,
        float((pred[ben] == 1).mean()) if ben.any() else 0.0,
    )


def choose_thresholds(
    p_s: Arr,
    p_t: Arr,
    y: NDArray[np.int64],
    teacher_thr: float,
    max_escalation: float = 0.15,
    asr_slack: float = 0.003,
    fpr_slack: float = 0.003,
    traffic: NDArray[np.float64] | None = None,
) -> Thresholds | None:
    """Grid search on val: the best-quality (t_low, t_high) with escalation under the cap.

    Feasible means cascade ASR and FPR within slack of the teacher's and escalation <= ``max_escalation``. Among
    feasible pairs the lowest ASR + FPR wins (ties: lower escalation), so the uncertain band is spent where the student
    is wrong. Candidates are the student's own score quantiles. ``traffic`` (optional) gives one weight per row: the
    escalation cap is the weighted escalation rate, so extra tuning attacks can join the pool and the benign/attack
    mix of real game traffic can be modelled without changing the pool. ``None`` when no pair is feasible.
    """
    teacher = (p_t >= teacher_thr).astype(int)
    atk, ben = y == 1, y == 0
    base_asr = float((teacher[atk] == 0).mean())
    base_fpr = float((teacher[ben] == 1).mean())
    qs = np.unique(np.concatenate([np.linspace(0.0, 1.0, 401), [0.995, 0.999]]))
    cand = np.unique(np.quantile(p_s, qs))
    best: Thresholds | None = None
    for lo in cand:
        for hi in cand[cand >= lo]:
            m = cascade_metrics(p_s, p_t, y, teacher_thr, float(lo), float(hi))
            esc = m.escalation
            if traffic is not None:
                flag = ((p_s >= lo) & (p_s <= hi)).astype(np.float64)
                esc = float((flag * traffic).sum() / traffic.sum())
                m = Thresholds(m.t_low, m.t_high, esc, m.asr, m.fpr)
            if esc > max_escalation:
                continue
            if m.asr > base_asr + asr_slack or m.fpr > base_fpr + fpr_slack:
                continue
            if best is None or (m.asr + m.fpr, m.escalation) < (
                best.asr + best.fpr,
                best.escalation,
            ):
                best = m
    return best
