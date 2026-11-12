"""Entry point executed inside a Kaggle kernel / Modal GPU container: runs one training pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from breachlab.common.config import ServeConfig, TrainConfig


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-dpo", action="store_true")
    ap.add_argument(
        "--stage", default="train", choices=["train", "zeroshot", "deberta", "evalgold", "export"]
    )
    ap.add_argument("--adapters", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--which", default="dpo-0.6b-s0", help="export stage: adapter name or 'base'")
    ap.add_argument("--variants", default="fp32,w8", help="export stage: ONNX variants to build")
    ap.add_argument(
        "--parity-rows", type=int, default=-1, help="export stage: gold rows to score (-1 all)"
    )
    return ap.parse_args(argv)


def config_from(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(
        base_model=args.model, seed=args.seed, data_dir=args.data_dir, output_dir=args.out
    )
    cfg.sft.precision = args.precision
    return cfg


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - GPU
    from breachlab.train.pipeline import run, run_deberta, run_evalgold, run_export, run_zeroshot

    args = parse(argv if argv is not None else sys.argv[1:])
    if args.stage == "deberta":
        print(json.dumps({"seconds": run_deberta(config_from(args))["seconds"]}))
        return 0
    if args.stage == "export":
        res = run_export(
            config_from(args),
            ServeConfig(),
            args.adapters,
            args.which,
            tuple(args.variants.split(",")),
            args.parity_rows,
        )
        print(json.dumps(res, default=str))
        return 0
    if args.stage == "evalgold":
        print(
            json.dumps(run_evalgold(config_from(args), ServeConfig(), args.adapters), default=str)
        )
        return 0
    if args.stage == "zeroshot":
        print(json.dumps(run_zeroshot(config_from(args), ServeConfig()), default=str))
        return 0
    res = run(config_from(args), ServeConfig(), with_dpo=not args.no_dpo)
    print(json.dumps({k: v for k, v in res.items() if k != "final_adapter"}, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
