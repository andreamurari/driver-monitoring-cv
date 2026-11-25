"""CPU latency, batch 1: the time for a single frame, as in a car.

Usage:
    python -m dms.benchmark
    python -m dms.benchmark --models resnet18 vit_s16 --sizes 224 --threads 4
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from dms.models.classifier import build_model, count_parameters


def measure_latency(model, size: int, runs: int = 100, warmup: int = 20) -> dict[str, float]:
    """Latency in milliseconds on a random 1 x 3 x size x size input.

    Returns {"median_ms": ..., "p95_ms": ..., "mean_ms": ...}.
    """
    model.eval()  # BatchNorm with stored statistics, dropout off
    x = torch.randn(1, 3, size, size)  # fake input: only the shape matters, not the content
    times = []
    with torch.inference_mode():  # no gradients
        for _ in range(warmup):  # the first runs are slower: not measured
            model(x)
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)  # seconds -> milliseconds
    return {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet18", "resnet50", "vit_s16"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[224, 320])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--out", default="reports/latency.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)  # fixed threads: comparable measurements
    results = []
    for name in args.models:
        # no weights needed: latency depends on the architecture, not the weight values
        model = build_model(name, pretrained=False)
        total, _ = count_parameters(model)
        for size in args.sizes:
            if name.startswith("vit") and size != 224:
                print(f"{name} @ {size}: skipped (needs interpolated position embeddings)")
                continue
            lat = measure_latency(model, size, runs=args.runs)
            row = {"model": name, "size": size, "params_m": round(total / 1e6, 1), **lat}
            results.append(row)
            print(f"{name:10s} @ {size}: median {lat['median_ms']:.1f} ms, "
                  f"p95 {lat['p95_ms']:.1f} ms, {row['params_m']} M parameters")

    env = {"cpu": platform.processor(), "threads": args.threads, "torch": torch.__version__,
           "batch": 1, "runs": args.runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"env": env, "results": results}, indent=2))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
