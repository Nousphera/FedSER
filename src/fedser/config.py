"""Experiment parameters shared by the launcher and native Flower apps."""

import math
from pathlib import Path

METHODS = ("fedser", "fedavg", "centralized")


def defaults():
    return {
        "method": "fedser", "device": "cpu", "protocol": "session", "fold": 1,
        "partition": "random", "num-clients": 0, "num-rounds": 100, "participation": .8,
        "local-epochs": 1, "batch-size": 64, "learning-rate": .001,
        "labeled-fraction": .1, "quantity-variation": .25,
        "beta": .5, "temperature": 2., "delta": .5,
        "threshold-start": .5, "threshold-end": .9, "seed": 2021, "data-seed": 2021,
        "data-root": "data", "output": "results/fedser.json",
        "torch-threads": 1, "num-workers": 0, "registration-timeout": 120., "reply-timeout": 86400.,
    }


def validate(cfg):
    cfg = dict(cfg)
    if set(cfg) != set(defaults()):
        raise ValueError(f"Unexpected or missing configuration keys: {set(cfg) ^ set(defaults())}")
    for key, value in cfg.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if cfg["method"] not in METHODS or cfg["protocol"] not in {"session", "overlap"} or cfg["partition"] not in {"random", "speaker"}:
        raise ValueError("Unknown method, evaluation protocol or client partition")
    for key in ("num-rounds", "local-epochs", "batch-size", "torch-threads"):
        if type(cfg[key]) != int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("num-clients", "seed", "data-seed", "num-workers"):
        if type(cfg[key]) != int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if type(cfg["fold"]) != int or cfg["fold"] not in range(1, 6):
        raise ValueError("fold must be 1..5")
    if cfg["method"] == "fedser" and cfg["labeled-fraction"] < 1 and (cfg["batch-size"] < 2 or cfg["batch-size"] % 2):
        raise ValueError("FedSER batch-size must be even: half labeled, half unlabeled")
    for key in ("labeled-fraction", "participation"):
        if not 0 < cfg[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if not 0 <= cfg["quantity-variation"] < 1 or not 0 <= cfg["delta"] <= 1:
        raise ValueError("quantity-variation must be in [0, 1); delta must be in [0, 1]")
    if not 0 <= cfg["threshold-start"] <= cfg["threshold-end"] <= 1:
        raise ValueError("Confidence thresholds must satisfy 0 <= start <= end <= 1")
    if cfg["beta"] < 0 or cfg["temperature"] <= 0 or cfg["learning-rate"] <= 0:
        raise ValueError("Invalid loss or optimizer parameters")
    if cfg["registration-timeout"] <= 0 or cfg["reply-timeout"] <= 0:
        raise ValueError("Timeouts must be positive")
    if cfg["device"] not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be cpu, mps or cuda; Flower assigns each client's GPU")
    if Path(cfg["output"]).suffix != ".json":
        raise ValueError("output must end in .json; model weights use a separate .pt file")
    return cfg
