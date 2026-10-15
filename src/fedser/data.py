"""Validated IEMOCAP manifests and deterministic, disjoint experiment partitions."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

CLASSES = ["neutral", "happy", "sad", "angry"]
EMOTIONS = {"neu": 0, "hap": 1, "sad": 2, "ang": 3}
UTTERANCE = re.compile(r"^Ses0([1-5])[FM]_impro\d+[a-z]?_([FM])\d+$")
PROVIDER = "https://sail.usc.edu/iemocap/"


@dataclass
class CorpusBundle:
    records: list
    classes: list
    fingerprint: str
    metadata: dict


@dataclass
class DataBundle:
    train: list
    test: list
    classes: list
    fingerprint: str
    metadata: dict = field(default_factory=dict)


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_fingerprint(value):
    payload = {k: v for k, v in value.items() if k != "fingerprint"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def safe_path(root, relative):
    """Resolve a canonical relative path without permitting a symlink escape."""
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("Expected a nonempty relative corpus path")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in relative.split("/")) or ":" in relative:
        raise ValueError(f"Audio path escapes corpus root or is not canonical: {relative}")
    resolved = (Path(root) / path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Audio path escapes corpus root: {relative}")
    return resolved


def _audio_metadata(path):
    import soundfile as sf
    try:
        with Path(path).open("rb") as stream:
            header = stream.read(12)
        if header[:4] == b"RIFF" and len(header) == 12 and int.from_bytes(header[4:8], "little") + 8 > Path(path).stat().st_size:
            raise ValueError(f"Truncated RIFF audio: {path}")
        with sf.SoundFile(path) as audio:
            if audio.format not in {"WAV", "WAVEX"} or audio.samplerate != 16000 or audio.channels != 1:
                raise ValueError(f"Expected mono 16000 Hz WAV: {path}")
            if audio.frames < 1:
                raise ValueError(f"Empty audio: {path}")
            frames = 0
            for block in audio.blocks(blocksize=65536, dtype="float32"):
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite audio: {path}")
                frames += len(block)
            if frames != audio.frames:
                raise ValueError(f"Truncated audio: {path}")
            return {"sample_rate": audio.samplerate, "frames": frames, "channels": audio.channels}
    except RuntimeError as error:
        raise ValueError(f"Invalid or corrupt audio: {path}: {error}") from error


def check_records(records, root, *, populate=False, verify=False):
    """Check all five sessions, metadata, file ownership, and stored integrity data."""
    if not isinstance(records, list) or not records:
        raise ValueError("IEMOCAP requires nonempty records")
    root = Path(root).resolve()
    ids, files, inodes, hashes = set(), set(), set(), set()
    output = []
    for original in records:
        if not isinstance(original, dict):
            raise ValueError("Invalid record")
        row = dict(original)
        uid = row.get("id")
        match = UTTERANCE.fullmatch(uid) if isinstance(uid, str) else None
        if not match or uid in ids:
            raise ValueError(f"Invalid or duplicate improvised utterance ID: {uid}")
        ids.add(uid)
        session, speaker = int(match[1]), f"Ses0{match[1]}{match[2]}"
        if type(row.get("session")) is not int or row["session"] != session or row.get("speaker") != speaker:
            raise ValueError(f"Session/speaker metadata disagrees with utterance ID: {uid}")
        label = row.get("label")
        if type(label) is not int or not 0 <= label < len(CLASSES):
            raise ValueError(f"Invalid label: {uid}")
        if row.get("class", CLASSES[label]) != CLASSES[label]:
            raise ValueError(f"Class and label disagree: {uid}")
        if row.get("emotion") not in EMOTIONS | {"exc": 1} or (EMOTIONS | {"exc": 1})[row["emotion"]] != label:
            raise ValueError(f"Emotion and label disagree: {uid}")
        expected = f"Session{session}/sentences/wav/{uid.rsplit('_', 1)[0]}/{uid}.wav"
        path = safe_path(root, row.get("path"))
        if row["path"] != expected:
            raise ValueError(f"Audio path does not match utterance ID: {uid}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing IEMOCAP audio: {path}")
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if path in files or inode in inodes:
            raise ValueError(f"Duplicate file or ownership leakage: {path}")
        files.add(path)
        inodes.add(inode)
        if not populate:
            if type(row.get("bytes")) is not int or row["bytes"] != stat.st_size:
                raise ValueError(f"Audio size changed or missing integrity metadata: {path}")
            if not isinstance(row.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
                raise ValueError(f"Invalid audio checksum: {path}")
            if (type(row.get("frames")) is not int or row["frames"] < 1
                    or row.get("sample_rate") != 16000 or row.get("channels") != 1):
                raise ValueError(f"Invalid audio metadata: {path}")
        if populate or verify:
            metadata = _audio_metadata(path)
            digest = checksum(path)
            if not populate and (row["sha256"] != digest or any(row[k] != v for k, v in metadata.items())):
                raise ValueError(f"Audio checksum or decoded metadata changed: {path}")
            row.update(metadata, sha256=digest, bytes=stat.st_size)
        if row["sha256"] in hashes:
            raise ValueError(f"Duplicate audio bytes risk ownership leakage: {path}")
        hashes.add(row["sha256"])
        row["class"] = CLASSES[label]
        output.append(row)
    if {r["session"] for r in output} != set(range(1, 6)):
        raise ValueError("Complete IEMOCAP requires all five sessions")
    if {r["speaker"] for r in output} != {f"Ses0{s}{g}" for s in range(1, 6) for g in "FM"}:
        raise ValueError("Complete IEMOCAP requires all ten speakers")
    if {r["label"] for r in output} != set(range(4)):
        raise ValueError("IEMOCAP requires all four emotion classes")
    return sorted(output, key=lambda r: r["id"])


def manifest_path(data_root="data"):
    root = Path(data_root).expanduser().resolve()
    if root.suffix == ".json":
        return root
    return root / "manifest.json" if root.name == "iemocap" else root / "iemocap/manifest.json"


def read_manifest(path="data/iemocap/manifest.json", *, verify=False):
    path = manifest_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"IEMOCAP is not prepared at {path}. Obtain licensed data from {PROVIDER}, "
                                "then run python scripts/prepare_data.py --source-dir PATH.")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != 1 or value.get("dataset") != "iemocap":
        raise ValueError(f"Wrong IEMOCAP manifest schema: {path}")
    if value.get("classes") != CLASSES:
        raise ValueError("Manifest class order differs from IEMOCAP class mapping")
    if value.get("fingerprint") != manifest_fingerprint(value):
        raise ValueError("Manifest fingerprint changed; metadata was modified")
    root = value.get("root")
    if not isinstance(root, str) or not root or Path(root).is_absolute():
        raise ValueError("Manifest corpus root must be relative to its directory")
    root = (path.parent / root).resolve()
    preparation = value.get("preparation", {})
    if type(preparation.get("merge_excited")) is not bool or preparation.get("selection") != "improvised-four-emotions":
        raise ValueError("Invalid IEMOCAP preparation protocol")
    rows = check_records(value.get("records"), root, verify=verify)
    if not preparation["merge_excited"] and any(r["emotion"] == "exc" for r in rows):
        raise ValueError("Excited annotation present under strict happy protocol")
    if value.get("counts") != record_counts(rows):
        raise ValueError("Manifest counts disagree with records")
    annotations = value.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("Missing source annotation integrity metadata")
    seen = set()
    expected_annotations = {p.resolve() for p in root.glob("Session[1-5]/dialog/EmoEvaluation/*.txt")}
    for item in annotations:
        if not isinstance(item, dict):
            raise ValueError("Invalid source annotation metadata")
        source = safe_path(root, item.get("path"))
        if source in seen or not source.is_file():
            raise ValueError(f"Missing or duplicate source annotation: {source}")
        seen.add(source)
        if item.get("bytes") != source.stat().st_size or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise ValueError(f"Annotation metadata changed: {source}")
        if verify and checksum(source) != item["sha256"]:
            raise ValueError(f"Annotation checksum changed: {source}")
    if seen != expected_annotations:
        raise ValueError("Source annotation inventory changed; prepare a new data root")
    return CorpusBundle([{**r, "path": str(safe_path(root, r["path"]))} for r in rows],
                        list(CLASSES), value["fingerprint"], preparation)


def record_counts(records):
    return {"utterances": len(records),
            "classes": {c: sum(r["label"] == i for r in records) for i, c in enumerate(CLASSES)},
            "sessions": {str(s): sum(r["session"] == s for r in records) for s in range(1, 6)},
            "speakers": {s: sum(r["speaker"] == s for r in records) for s in sorted({r["speaker"] for r in records})}}


def _seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def _unique(records):
    ids = [r.get("id") for r in records]
    if not records or any(not isinstance(uid, str) or not uid for uid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Empty records or duplicate utterance ownership")
    if any(type(r.get("label")) is not int or not 0 <= r["label"] < 4 for r in records):
        raise ValueError("Expected original four-class labels before partitioning")
    return sorted(records, key=lambda r: r["id"])


def split_records(records, protocol, fold, seed):
    records = _unique(records)
    _seed(seed)
    if type(fold) is not int or fold not in range(1, 6):
        raise ValueError("fold must be 1..5")
    if protocol == "session":
        if {r.get("session") for r in records} != set(range(1, 6)):
            raise ValueError("Session folds require all five sessions")
        test_ids = {r["id"] for r in records if r["session"] == fold}
    elif protocol == "overlap":
        rng, cursor, test_ids = np.random.default_rng(seed), 0, set()
        for label in range(4):
            indices = [i for i, row in enumerate(records) if row["label"] == label]
            if len(indices) < 5:
                raise ValueError("Five-fold overlap requires at least five utterances per class")
            for offset, index in enumerate(rng.permutation(indices)):
                if (cursor + offset) % 5 == fold - 1:
                    test_ids.add(records[index]["id"])
            cursor = (cursor + len(indices)) % 5
    else:
        raise ValueError("protocol must be session or overlap")
    train = [r for r in records if r["id"] not in test_ids]
    test = [r for r in records if r["id"] in test_ids]
    if not train or not test:
        raise ValueError("Empty train or test fold")
    return train, test


def partition_clients(records, partition, clients, variation=.25, seed=2021):
    records = _unique(records)
    _seed(seed)
    if type(clients) is not int or not 1 <= clients <= len(records) or not 0 <= variation < 1:
        raise ValueError("Invalid client count or quantity variation")
    if partition == "speaker":
        speakers = sorted({r["speaker"] for r in records})
        if len(speakers) != clients:
            raise ValueError(f"Speaker partition needs {len(speakers)} actual clients, received {clients}")
        return [[r for r in records if r["speaker"] == s] for s in speakers]
    if partition != "random":
        raise ValueError("partition must be random or speaker")
    rng = np.random.default_rng(seed)
    mean = len(records) / clients
    low, high = max(1, int(np.floor(mean * (1 - variation)))), int(np.ceil(mean * (1 + variation)))
    sizes = np.full(clients, low, dtype=int)
    for _ in range(len(records) - int(sizes.sum())):
        sizes[rng.choice(np.flatnonzero(sizes < high))] += 1
    order = rng.permutation(len(records))
    return [[records[i] for i in group] for group in np.split(order, np.cumsum(sizes)[:-1])]


def label_clients(clients, fraction, seed, *, partition="random"):
    """Select labels globally, then apply the same mask to fixed client ownership.

    Random: floor(fraction * class count) per class. Speaker: the original joint
    speaker/class StratifiedShuffleSplit allocation with floor(fraction * N).
    No extra labels are silently added to repair an empty client label set.
    """
    _seed(seed)
    if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError("labeled-fraction must be in (0, 1]")
    records = _unique([r for client in clients for r in client])
    if any(not client for client in clients):
        raise ValueError("Every client must own data")
    if partition not in {"random", "speaker"}:
        raise ValueError("partition must be random or speaker")
    if fraction == 1:
        visible = {r["id"] for r in records}
    elif partition == "speaker":
        from sklearn.model_selection import train_test_split
        strata = [f"{r['speaker']}:{r['label']}" for r in records]
        try:
            selected, _ = train_test_split(np.arange(len(records)), train_size=fraction,
                                           random_state=seed, stratify=strata)
        except ValueError as error:
            raise ValueError(f"Speaker/class label stratification cannot meet this label budget: {error}") from error
        visible = {records[i]["id"] for i in selected}
    else:
        rng, visible = np.random.default_rng(seed), set()
        for label in range(4):
            rows = [r for r in records if r["label"] == label]
            visible.update(rows[i]["id"] for i in rng.permutation(len(rows))[:int(len(rows) * fraction)])
    output = []
    for cid, rows in enumerate(clients):
        if not any(r["id"] in visible for r in rows):
            raise ValueError(f"Client {cid} has no labeled utterances at this budget; increase labeled-fraction "
                             "or reduce num-clients (the global label budget is not enlarged automatically)")
        # Do not retain any alternative spelling of the ground-truth emotion.
        output.append([{k: v for k, v in r.items() if k not in {"class", "emotion"}} |
                       {"label": r["label"] if r["id"] in visible else -1} for r in rows])
    return output


def load_data(data_root="data", cfg=None, verify_audio=False):
    cfg = cfg or {}
    corpus = read_manifest(manifest_path(data_root), verify=verify_audio)
    train, test = split_records(corpus.records, cfg.get("protocol", "session"),
                               cfg.get("fold", 1), cfg.get("data-seed", 2021))
    return DataBundle(train, test, corpus.classes, corpus.fingerprint, corpus.metadata)


def _ownership_count(train, cfg):
    count = cfg.get("num-clients", 0)
    if type(count) is not int or count < 0:
        raise ValueError("num-clients must be a nonnegative integer")
    return count or (len({r["speaker"] for r in train}) if cfg.get("partition", "random") == "speaker" else 10)


def resolve_clients(train, cfg):
    return 1 if cfg.get("method", "fedser") == "centralized" else _ownership_count(train, cfg)


def partition_records(train, cfg):
    method = cfg.get("method", "fedser")
    if method not in {"fedser", "fedavg", "centralized"}:
        raise ValueError("method must be fedser, fedavg or centralized")
    clients = partition_clients(train, cfg.get("partition", "random"), _ownership_count(train, cfg),
                                cfg.get("quantity-variation", .25), cfg.get("data-seed", 2021))
    clients = label_clients(clients, cfg.get("labeled-fraction", .1), cfg.get("data-seed", 2021),
                            partition=cfg.get("partition", "random"))
    if method in {"fedavg", "centralized"}:
        clients = [[r for r in client if r["label"] >= 0] for client in clients]
    return [[r for client in clients for r in client]] if method == "centralized" else clients
