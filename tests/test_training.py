import json
import numpy as np
import pytest
import tensorflow as tf
from fedser.data import read_iemocap
from fedser.train import RunConfig, make_data, local_train, run
from fedser.model import build_model


def test_local_fit_changes_parameters_and_uses_exact_counts():
    config = RunConfig(clients=2, rounds=1, batch_size=4, synthetic_per_client=8, labeled_fraction=.5)
    clients, test, shape = make_data(config)
    model = build_model(shape)
    weights = model.get_weights()
    result, count, metrics = local_train(weights, clients[0], shape, config, 0, 0, 0)
    assert count == 8
    assert metrics["steps"] == 1
    assert metrics["unlabeled"] == 4
    assert np.isfinite(metrics["loss"])
    assert any(not np.array_equal(a, b) for a, b in zip(weights, result))


def test_end_to_end_training_writes_experiment_artifacts(tmp_path):
    config = RunConfig(clients=2, rounds=2, participation=1., batch_size=4,
                       synthetic_per_client=8, labeled_fraction=.5, output=str(tmp_path / "run"))
    result = run(config)
    assert result["dataset"] == "synthetic" and result["mode"] == config.mode
    assert len(result["rounds"]) == 2
    assert all(r["selected_clients"] == [0, 1] for r in result["rounds"])
    assert all(0 <= r["ua"] <= 1 for r in result["rounds"])
    assert (tmp_path / "run/model.keras").is_file()
    assignment = json.loads((tmp_path / "run/partitions.json").read_text())
    train_ids = {r["id"] for c in assignment["clients"] for r in c}
    assert not train_ids & set(assignment["test_ids"])
    with pytest.raises(FileExistsError):
        run(config)


def test_rejects_invalid_config_and_supervised_baseline_discards_unlabeled():
    with pytest.raises(ValueError):
        RunConfig(participation=0)
    config = RunConfig(clients=2, synthetic_per_client=8, labeled_fraction=.5, supervised_only=True)
    clients, _, _ = make_data(config)
    assert [len(c) for c in clients] == [4, 4]
    assert all(r["label"] >= 0 for c in clients for r in c)


def test_iemocap_adapter_to_real_wav_training(tmp_path):
    root = tmp_path / "corpus"
    for session in (1, 2):
        lines = []
        for speaker in "FM":
            for label, emotion in enumerate(("neu", "hap", "sad", "ang")):
                for sample in range(2):
                    uid = f"Ses{session:02}F_impro01_{speaker}{label * 2 + sample:03}"
                    path = root / f"Session{session}/sentences/wav" / uid.rsplit("_", 1)[0] / f"{uid}.wav"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    wave = (.2 * np.sin(np.arange(32000) * (.02 + .01 * label))).astype(np.float32)
                    path.write_bytes(tf.audio.encode_wav(wave[:, None], 16000).numpy())
                    lines.append(f"[0.0 - 2.0]\t{uid}\t{emotion}\t[2, 2, 2]")
        annotation = root / f"Session{session}/dialog/EmoEvaluation/test.txt"
        annotation.parent.mkdir(parents=True)
        annotation.write_text("\n".join(lines))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"root": str(root), "records": read_iemocap(root)}))
    config = RunConfig(dataset="iemocap", manifest=str(manifest), clients=2, partition="speaker",
                       fold=1, rounds=1, participation=1., labeled_fraction=.5,
                       batch_size=4, output=str(tmp_path / "run"))
    result = run(config)
    assert len(result["rounds"]) == 1
    assert all(m["num_examples"] == 8 for m in result["rounds"][0]["client_metrics"])
