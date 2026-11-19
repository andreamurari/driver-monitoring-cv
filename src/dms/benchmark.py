"""Latenza su CPU, batch 1: il tempo per un singolo fotogramma, come in auto.

Uso:
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
    """Latenza in millisecondi su un input casuale 1 x 3 x size x size.

    Restituisce {"median_ms": ..., "p95_ms": ..., "mean_ms": ...}.
    """
    model.eval()  # BatchNorm con le statistiche salvate, dropout spento
    x = torch.randn(1, 3, size, size)  # input finto: conta la forma, non il contenuto
    times = []
    with torch.inference_mode():  # niente gradienti
        for _ in range(warmup):  # le prime esecuzioni sono più lente: non si misurano
            model(x)
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)  # secondi -> millisecondi
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

    torch.set_num_threads(args.threads)  # thread fissi: misure confrontabili
    results = []
    for name in args.models:
        # pesi non necessari: la latenza dipende dall'architettura, non dai valori dei pesi
        model = build_model(name, pretrained=False)
        total, _ = count_parameters(model)
        for size in args.sizes:
            if name.startswith("vit") and size != 224:
                print(f"{name} @ {size}: saltato (servono embedding di posizione interpolati)")
                continue
            lat = measure_latency(model, size, runs=args.runs)
            row = {"model": name, "size": size, "params_m": round(total / 1e6, 1), **lat}
            results.append(row)
            print(f"{name:10s} @ {size}: mediana {lat['median_ms']:.1f} ms, "
                  f"p95 {lat['p95_ms']:.1f} ms, {row['params_m']} M parametri")

    env = {"cpu": platform.processor(), "threads": args.threads, "torch": torch.__version__,
           "batch": 1, "runs": args.runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"env": env, "results": results}, indent=2))
    print(f"salvato in {args.out}")


if __name__ == "__main__":
    main()
