"""Launch experiments with the installed native Flower CLI."""

import argparse
from contextlib import contextmanager
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import torch
from .config import METHODS, defaults, validate


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, default="fedser")
    p.add_argument("--protocol", choices=("session", "overlap"), default="session")
    p.add_argument("--partition", choices=("random", "speaker"), default="random")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--nr-clients", type=int, default=0, help="Client count; default: 10 random clients or one per training speaker")
    p.add_argument("--rounds", type=int, default=100)
    for key in ("local-epochs", "batch-size", "seed", "data-seed", "torch-threads", "num-workers"):
        p.add_argument(f"--{key}", type=int, default=defaults()[key])
    p.add_argument("--labeled-fraction", type=float)
    for key in ("participation", "quantity-variation", "delta", "learning-rate",
                "beta", "temperature", "threshold-start", "threshold-end", "registration-timeout", "reply-timeout"):
        p.add_argument(f"--{key}", type=float, default=defaults()[key])
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path)
    p.add_argument("--parallel-clients", type=int, default=1)
    p.add_argument("--client-cpus", type=int, default=1)
    p.add_argument("--client-gpus", type=float)
    p.add_argument("--superlink", default="local")
    p.add_argument("--dry-run", action="store_true", help="Print resolved configuration without starting training")
    return p


def build_config(args):
    cfg = defaults()
    for key in cfg:
        attribute = {"num-clients": "nr_clients", "num-rounds": "rounds"}.get(key, key.replace("-", "_"))
        value = getattr(args, attribute, None)
        if value is not None:
            cfg[key] = value
    if args.labeled_fraction is None and args.method == "centralized":
        cfg["labeled-fraction"] = 1.
    cfg["data-root"] = str(cfg["data-root"].expanduser().resolve())
    name = (f"{args.method}-{args.protocol}-{args.partition}-fold{args.fold}"
            f"-l{100 * cfg['labeled-fraction']:g}-seed{args.seed}.json")
    cfg["output"] = str((args.output or Path("results") / name).expanduser().resolve())
    if args.parallel_clients < 1 or args.client_cpus < 1:
        raise ValueError("Client resources must be positive")
    return validate(cfg)


@contextmanager
def flower_app(cfg):
    """Build Flower's descriptor temporarily, keeping one installation manifest."""
    import tomli_w

    dist = distribution("fedser")
    package = Path(__file__).resolve().parent
    assets = package.parent.parent
    if not (assets / "setup.py").exists():
        assets = Path(sys.prefix) / "share/fedser"
    descriptor = {
        "build-system": {"requires": ["setuptools>=77", "wheel"], "build-backend": "setuptools.build_meta"},
        "project": {"name": "fedser", "version": dist.version, "description": dist.metadata["Summary"],
                    "requires-python": dist.metadata["Requires-Python"], "dependencies": dist.requires or [],
                    "readme": {"text": dist.metadata["Summary"], "content-type": "text/plain"}},
        "tool": {"flwr": {"app": {"publisher": "nousphera", "components": {
            "serverapp": "fedser.server_app:app", "clientapp": "fedser.client_app:app"}, "config": cfg}}},
    }
    with TemporaryDirectory(prefix="fedser-flower-") as directory:
        root = Path(directory)
        for source in package.rglob("*"):
            if source.is_file() and source.suffix in {".py", ".txt"} and "__pycache__" not in source.parts:
                target = root / "src/fedser" / source.relative_to(package)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        for name in ("setup.py",):
            shutil.copy2(assets / name, root / name)
        (root / "pyproject.toml").write_text(tomli_w.dumps(descriptor))
        yield root


def flower_command(args, cfg, config_file, application, nodes=None):
    executable = Path(sys.executable).with_name("flwr")
    if not executable.exists():
        raise RuntimeError("Install requirements.txt in the active environment first")
    gpus = args.client_gpus if args.client_gpus is not None else (1. if cfg["device"].startswith("cuda") else 0.)
    import math
    if not math.isfinite(gpus) or gpus < 0 or (cfg["device"].startswith("cuda") and gpus <= 0):
        raise ValueError("CUDA clients require positive --client-gpus")
    if nodes is None:
        from .data import load_data, partition_records
        nodes = len(partition_records(load_data(cfg["data-root"], cfg).train, cfg))
    federation = (f"num-supernodes={nodes} client-resources-num-cpus={args.client_cpus} "
                  f"client-resources-num-gpus={gpus} init-args-num-cpus={args.client_cpus * args.parallel_clients}")
    return [str(executable), "run", str(application), args.superlink, "--run-config", str(config_file),
            "--federation-config", federation, "--stream"]


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        cfg = build_config(args)
        path = Path(cfg["output"])
        config_file = path.with_suffix(".run.toml")
        if args.dry_run:
            with flower_app(cfg) as application:
                print(json.dumps({"config": cfg, "command": flower_command(args, cfg, config_file, application)}, indent=2))
            return
        from .data import load_data, partition_records
        bundle = load_data(cfg["data-root"], cfg, verify_audio=True)
        nodes = len(partition_records(bundle.train, cfg))
        if cfg["device"].startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; choose --device cpu or --device mps")
        if cfg["device"] == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable; choose --device cpu")
        for target in (path, path.with_suffix(".pt"), config_file):
            if target.exists():
                raise FileExistsError(f"Output exists: {target}")
        path.parent.mkdir(parents=True, exist_ok=True)
        import tomli_w
        with config_file.open("xb") as stream:
            tomli_w.dump(cfg, stream)
        env = os.environ.copy()
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        env.setdefault("FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION", "1")
        with flower_app(cfg) as application:
            command = flower_command(args, cfg, config_file, application, nodes)
            print(shlex.join(command), flush=True)
            subprocess.run(command, check=True, env=env)
        if not path.exists() or not path.with_suffix(".pt").exists():
            raise RuntimeError("Flower ended without saved results; inspect the run logs")
        print(f"Saved {path}")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.exit(2, f"Experiment failed: {exc}\n")


if __name__ == "__main__":
    main()
