"""Prepare the licensed IEMOCAP corpus without downloading or modifying audio."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .data import (CLASSES, EMOTIONS, PROVIDER, UTTERANCE, check_records, checksum,
                   manifest_fingerprint, read_manifest, record_counts, safe_path)

SUMMARY = re.compile(r"^\[\s*([\d.]+)\s*-\s*([\d.]+)\s*\]\s+(\S+)\s+(\S+)\s+\[([^]]+)\]\s*$")


def annotation_paths(root):
    root = Path(root).expanduser().resolve()
    paths = []
    for session in range(1, 6):
        directory = root / f"Session{session}/dialog/EmoEvaluation"
        found = sorted(directory.glob("*.txt"))
        if not found:
            raise FileNotFoundError(f"Missing IEMOCAP Session{session} annotations at {directory}. "
                                    f"Obtain the corpus from {PROVIDER} and extract all five sessions.")
        paths.extend(found)
    return paths


def read_iemocap(root, merge_excited=False):
    """Read consensus annotations, excluding scripts and unselected emotions."""
    if type(merge_excited) is not bool:
        raise ValueError("merge_excited must be boolean")
    root = Path(root).expanduser().resolve()
    emotions = EMOTIONS | ({"exc": 1} if merge_excited else {})
    records, seen = [], set()
    for annotation in annotation_paths(root):
        safe_path(root, annotation.relative_to(root).as_posix())
        session_folder = int(annotation.relative_to(root).parts[0].removeprefix("Session"))
        for number, line in enumerate(annotation.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line.startswith("[") or line.startswith("[START_TIME"):
                continue
            match = SUMMARY.fullmatch(line)
            if not match:
                raise ValueError(f"Malformed consensus annotation at {annotation}:{number}")
            start, stop, uid, emotion, _ = match.groups()
            if not math.isfinite(float(start)) or not math.isfinite(float(stop)) or float(stop) <= float(start):
                raise ValueError(f"Invalid annotation timestamps at {annotation}:{number}")
            if uid in seen:
                raise ValueError(f"Duplicate utterance annotation: {uid}")
            seen.add(uid)
            if "_impro" not in uid:
                continue
            identity = UTTERANCE.fullmatch(uid)
            if not identity or int(identity[1]) != session_folder:
                raise ValueError(f"Invalid improvised utterance/session at {annotation}:{number}: {uid}")
            if emotion not in EMOTIONS | {"exc": 1, "fru": -1, "fea": -1, "sur": -1, "dis": -1, "oth": -1, "xxx": -1}:
                raise ValueError(f"Unknown IEMOCAP emotion at {annotation}:{number}: {emotion}")
            if emotion not in emotions:
                continue
            session, speaker = int(identity[1]), f"Ses0{identity[1]}{identity[2]}"
            relative = f"Session{session}/sentences/wav/{uid.rsplit('_', 1)[0]}/{uid}.wav"
            if not safe_path(root, relative).is_file():
                raise FileNotFoundError(root / relative)
            records.append({"id": uid, "session": session, "speaker": speaker,
                            "emotion": emotion, "label": emotions[emotion], "path": relative})
    for audio in root.glob("Session[1-5]/sentences/wav/*_impro*/*.wav"):
        if audio.stem not in seen:
            raise ValueError(f"Missing consensus annotation for improvised audio: {audio}")
    if not records:
        raise ValueError("No four-class improvised utterances found in IEMOCAP annotations")
    return sorted(records, key=lambda row: row["id"])


def write_json(path, value):
    """Atomically publish once; never replace a pre-existing manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_dataset(data_root="data", *, source_dir=None, merge_excited=False, validate=False):
    """Validate every selected recording before publishing data/iemocap/manifest.json."""
    if type(merge_excited) is not bool:
        raise ValueError("merge_excited must be boolean")
    data_root = Path(data_root).expanduser().resolve()
    destination = data_root / "iemocap/manifest.json"
    source = Path(source_dir).expanduser().resolve() if source_dir is not None else data_root / "raw/iemocap"
    if destination.exists() or validate:
        bundle = read_manifest(destination, verify=True)
        if bundle.metadata["merge_excited"] != merge_excited:
            raise ValueError("IEMOCAP already prepared with a different merge-excited choice; choose a new --data-root")
        if source_dir is not None:
            previous_root = (destination.parent / json.loads(destination.read_text())["root"]).resolve()
            if previous_root != source:
                raise ValueError("IEMOCAP already prepared from another source directory; choose a new --data-root")
        return bundle
    if not source.is_dir():
        raise FileNotFoundError(f"Licensed IEMOCAP corpus not found at {source}. Request access at {PROVIDER}, "
                                "extract Session1 through Session5 under data/raw/iemocap, or run "
                                "python scripts/prepare_data.py --source-dir /path/to/IEMOCAP_full_release. "
                                "This command does not download restricted data or use substitute datasets.")
    rows = check_records(read_iemocap(source, merge_excited), source, populate=True)
    annotations = [{"path": path.relative_to(source).as_posix(), "bytes": path.stat().st_size,
                    "sha256": checksum(path)} for path in annotation_paths(source)]
    value = {"schema": 1, "dataset": "iemocap", "classes": list(CLASSES),
             "root": Path(os.path.relpath(source, destination.parent)).as_posix(),
             "preparation": {"selection": "improvised-four-emotions", "merge_excited": merge_excited,
                             "provider": PROVIDER, "sample_rate": 16000,
                             "label_note": "Excited is excluded unless --merge-excited is explicitly supplied."},
             "counts": record_counts(rows), "annotations": annotations, "records": rows}
    value["fingerprint"] = manifest_fingerprint(value)
    write_json(destination, value)
    return read_manifest(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare and verify all five sessions of a licensed IEMOCAP corpus.")
    parser.add_argument("--source-dir", type=Path, help="Corpus directory containing Session1/ through Session5/; default data/raw/iemocap")
    parser.add_argument("--data-root", default="data", help="Prepared manifest location: DATA_ROOT/iemocap/manifest.json")
    parser.add_argument("--merge-excited", action="store_true", help="Explicitly merge excited into happy; strict happy is the default")
    parser.add_argument("--validate", action="store_true", help="Decode/hash the existing manifest without preparing a new one")
    args = parser.parse_args(argv)
    try:
        bundle = prepare_dataset(args.data_root, source_dir=args.source_dir,
                                 merge_excited=args.merge_excited, validate=args.validate)
    except (OSError, ValueError) as error:
        print(f"iemocap: {error}")
        return 1
    print(f"iemocap: ready ({len(bundle.records)} utterances; 5 sessions; 10 speakers; 4 classes)")
    print(f"Manifest: {Path(args.data_root).resolve() / 'iemocap/manifest.json'}")
    return 0
