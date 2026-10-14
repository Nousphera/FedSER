"""Command-line entry points. Importing the package never downloads data."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="FedSER data preparation and training")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Index a licensed IEMOCAP corpus")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=Path("data/iemocap.json"))
    prepare.add_argument("--merge-excited", action="store_true")
    for name in ("train", "flower"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/smoke.json")
        command.add_argument("--output")
        command.add_argument("--seed", type=int)
        command.add_argument("--fold", type=int)
    args = parser.parse_args()
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    if args.command == "prepare":
        from .data import read_iemocap
        records = read_iemocap(args.root, args.merge_excited)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump({"schema": 1, "root": str(args.root.resolve()), "merge_excited": args.merge_excited,
                       "records": records}, f, indent=2)
        print(f"Indexed {len(records)} utterances in {args.output}")
    else:
        from .train import load_config, run
        config = load_config(args.config, output=args.output, seed=args.seed, fold=args.fold)
        if args.command == "train":
            run(config)
        else:
            from .server_app import simulate
            simulate(config)


if __name__ == "__main__":
    main()
