"""IEMOCAP parsing, leakage-safe partitions and source-derived audio features."""
from pathlib import Path
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

EMOTIONS = {"neu": 0, "hap": 1, "sad": 2, "ang": 3}


def read_iemocap(root, merge_excited=False):
    root = Path(root).resolve()
    emotions = {**EMOTIONS, **({"exc": 1} if merge_excited else {})}
    records, seen = [], set()
    for annotation in sorted(root.glob("Session*/dialog/EmoEvaluation/*.txt")):
        for line in annotation.read_text().splitlines():
            match = re.match(r"^\[[^]]+\]\s+(Ses(\d\d)[FM]_impro\S+_([FM])\d+)\s+(\w+)\s", line)
            if not match:
                continue
            uid, session, speaker, emotion = match.groups()
            if emotion not in emotions:
                continue
            path = Path(f"Session{int(session)}") / "sentences/wav" / uid.rsplit("_", 1)[0] / f"{uid}.wav"
            if not (root / path).is_file():
                raise FileNotFoundError(root / path)
            if uid in seen:
                raise ValueError(f"Duplicate utterance annotation: {uid}")
            seen.add(uid)
            records.append(dict(id=uid, session=int(session), speaker=f"Ses{session}{speaker}",
                                label=emotions[emotion], path=path.as_posix()))
    if not records:
        raise ValueError("No IEMOCAP improvised utterances found; check corpus root")
    return records


def split_records(records, protocol, fold, seed):
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate utterance IDs")
    if fold not in range(1, 6):
        raise ValueError("fold must be 1..5")
    if protocol == "session":
        train = [r for r in records if r["session"] != fold]
        test = [r for r in records if r["session"] == fold]
    elif protocol == "overlap":
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
        train_ids, test_ids = list(splitter.split(records, [r["label"] for r in records]))[fold - 1]
        train, test = ([records[i] for i in indices] for indices in (train_ids, test_ids))
    else:
        raise ValueError("protocol must be session or overlap")
    if not train or not test:
        raise ValueError("Empty train or test fold")
    return train, test


def partition_clients(records, partition, clients, variation, seed):
    if clients < 1 or len(records) < clients or not 0 <= variation < 1:
        raise ValueError("Invalid client count/quantity variation")
    if partition == "speaker":
        speakers = sorted({r["speaker"] for r in records})
        if len(speakers) != clients:
            raise ValueError(f"Speaker partition needs {len(speakers)} clients, received {clients}")
        return [[r for r in records if r["speaker"] == s] for s in speakers]
    if partition != "random":
        raise ValueError("partition must be random or speaker")
    rng = np.random.default_rng(seed)
    mean = len(records) / clients
    low, high = max(1, int(np.floor(mean * (1 - variation)))), int(np.ceil(mean * (1 + variation)))
    sizes = np.full(clients, low, dtype=int)
    for _ in range(len(records) - sizes.sum()):
        eligible = np.flatnonzero(sizes < high)
        sizes[rng.choice(eligible)] += 1
    shuffled = rng.permutation(len(records))
    return [[records[i] for i in group] for group in np.split(shuffled, np.cumsum(sizes)[:-1])]


def label_clients(clients, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("labeled_fraction must be in (0,1]")
    masked = []
    for cid, rows in enumerate(clients):
        rng = np.random.default_rng(np.random.SeedSequence([seed, cid]))
        visible = set()
        for label in range(4):
            indices = [i for i, r in enumerate(rows) if r["label"] == label]
            visible.update(rng.permutation(indices)[:int(len(indices) * fraction)].tolist())
        if not visible:
            raise ValueError(f"Client {cid} has no labels at this fraction; increase data or label budget")
        masked.append([{**row, "label": row["label"] if i in visible else -1} for i, row in enumerate(rows)])
    return masked


def read_wave(path):
    waveform, rate = tf.audio.decode_wav(tf.io.read_file(str(path)), desired_channels=1)
    if int(rate) != 16000:
        raise ValueError(f"Expected 16000 Hz PCM WAV: {path}; got {int(rate)}")
    wave = waveform[:, 0].numpy()
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError(f"Invalid audio: {path}")
    return wave


def features(waveform, training, rng, mode="paper"):
    if mode not in {"paper", "source"}:
        raise ValueError("mode must be paper or source")
    wave = np.asarray(waveform, np.float32)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Expected finite, nonempty mono waveform")
    padding = max(32000 - len(wave), 0)
    wave = np.pad(wave, (padding // 2, padding - padding // 2))
    length = 32000 if mode == "paper" else 31380
    if training:
        wave = tf.math.l2_normalize(wave, epsilon=1e-9).numpy()
        start = rng.integers(0, len(wave) - length + 1)
        segments = wave[start:start + length]
    else:
        stride = length if mode == "paper" else 6276
        segments = tf.signal.frame(wave, length, stride, pad_end=False)
        segments = tf.math.l2_normalize(segments, axis=-1, epsilon=1e-9)
    magnitude = tf.abs(tf.signal.stft(segments, frame_length=400, frame_step=160, fft_length=1024))
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(64, 513, 16000, 60., 7800.)
    mel = tf.tensordot(magnitude, mel_matrix, 1)
    return tf.math.log(tf.clip_by_value(mel, 1e-5, 1e8))[..., None].numpy()
