from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from fedser.data import read_iemocap, split_records, partition_clients, label_clients, features


def records():
    return [dict(id=f"{s}-{p}-{c}-{n}", session=s, speaker=f"{s}{p}", label=c, path="unused")
            for s in range(1, 6) for p in "FM" for c in range(4) for n in range(10)]


def test_parser_keeps_only_improvised_four_classes_and_correct_utterance_speaker(tmp_path):
    d = tmp_path / "Session1/dialog/EmoEvaluation"
    d.mkdir(parents=True)
    ids = ["Ses01F_impro01_M000", "Ses01F_impro01_F001", "Ses01F_script01_F000"]
    labels = ["neu", "exc", "hap"]
    for uid in ids:
        p = tmp_path / "Session1/sentences/wav" / uid.rsplit("_", 1)[0] / (uid + ".wav")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fixture")
    (d / "labels.txt").write_text("\n".join(f"[0.0 - 1.0]\t{u}\t{l}\t[1, 2, 3]" for u, l in zip(ids, labels)))
    strict = read_iemocap(tmp_path)
    assert len(strict) == 1
    assert strict[0]["speaker"] == "Ses01M"
    merged = read_iemocap(tmp_path, merge_excited=True)
    assert len(merged) == 2 and merged[1]["label"] == 1


def test_splits_and_client_partitions_preserve_utterance_disjointness():
    train, test = split_records(records(), "session", 2, 13)
    assert {r["session"] for r in test} == {2}
    assert not {r["id"] for r in train} & {r["id"] for r in test}
    clients = partition_clients(train, "speaker", 8, .25, 13)
    assert all(len({r["speaker"] for r in c}) == 1 for c in clients)
    random = partition_clients(train, "random", 10, .25, 13)
    assert random == partition_clients(train, "random", 10, .25, 13)
    assert sorted(r["id"] for c in random for r in c) == sorted(r["id"] for r in train)
    folds = [split_records(records(), "overlap", f, 13)[1] for f in range(1, 6)]
    assert len({r["id"] for fold in folds for r in fold}) == len(records())
    assert sum(map(len, folds)) == len(records())


def test_label_mask_preserves_classes_and_never_exposes_unlabeled_targets():
    clients = partition_clients(records(), "speaker", 10, .25, 13)
    masked = label_clients(clients, .1, 13)
    assert all(sum(r["label"] >= 0 for r in c) == 4 for c in masked)
    assert all({r["label"] for r in c} == {-1, 0, 1, 2, 3} for c in masked)
    assert all(r["label"] >= 0 for c in clients for r in c)


def test_nonoverlap_audio_segmentation_and_spectrogram_shape():
    wave = np.sin(np.arange(64000) * .04).astype(np.float32)
    result = features(wave, training=False, rng=np.random.default_rng(0))
    assert result.shape == (2, 198, 64, 1)
    assert np.isfinite(result).all()
    short = features(np.zeros(100, np.float32), training=True, rng=np.random.default_rng(0))
    assert short.shape == (198, 64, 1) and np.isfinite(short).all()
