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
    validate = commands.add_parser("validate", help="Decode and verify an IEMOCAP manifest")
    validate.add_argument("--manifest", required=True, type=Path)
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
        from .preparation import check_records, write_json
        from .data import read_wave
        records = check_records(records, list(range(4)), None, args.root, read_wave)
        write_json(args.output, {"schema": 1, "root": str(args.root.resolve()),
                   "merge_excited": args.merge_excited, "records": records})
        print(f"Validated {len(records)} utterances in {args.output}")
    elif args.command == "validate":
        from .preparation import check_records
        from .data import read_wave
        value = json.loads(args.manifest.read_text())
        check_records(value["records"], list(range(4)), None, value["root"], read_wave, True)
        print(f"Validated {len(value['records'])} utterances")
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
