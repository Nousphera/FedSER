import json
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from fedser.preparation import check_records, write_json
from fedser.data import read_wave
from fedser.train import RunConfig, make_data


def test_atomic_json_failure_never_publishes_partial_output(tmp_path):
    output = tmp_path / "manifest.json"
    with pytest.raises(TypeError):
        write_json(output, {"invalid": object()})
    assert not output.exists() and list(tmp_path.iterdir()) == []
    write_json(output, {"valid": True})
    with pytest.raises(FileExistsError):
        write_json(output, {})
    assert json.loads(output.read_text()) == {"valid": True}


def test_decoded_manifest_rejects_tamper_and_path_escape(tmp_path):
    wave = tmp_path / "a.wav"
    tf.io.write_file(str(wave), tf.audio.encode_wav(tf.zeros((1600, 1)), 16000))
    row = {"id": "a", "path": "a.wav", "label": 0}
    checked = check_records([row], list(range(4)), None, tmp_path, read_wave)
    assert len(checked[0]["sha256"]) == 64
    tf.io.write_file(str(wave), tf.audio.encode_wav(tf.ones((1600, 1)), 16000))
    with pytest.raises(ValueError, match="checksum"):
        check_records(checked, list(range(4)), None, tmp_path, read_wave, True)
    with pytest.raises(ValueError, match="escapes"):
        check_records([{**row, "path": str(wave)}], list(range(4)), None, tmp_path)
    with pytest.raises(ValueError, match="Duplicate file"):
        check_records([row, {**row, "id": "b"}], list(range(4)), None, tmp_path)


def test_training_rejects_bad_manifest_labels(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"root": str(tmp_path), "records": [{"id": "x", "label": -1, "path": "missing.wav"}]}))
    with pytest.raises(ValueError, match="label"):
        make_data(RunConfig(dataset="iemocap", manifest=str(path)))


def test_config_paths_stay_with_checkout_when_working_directory_changes(tmp_path, monkeypatch):
    from fedser.train import load_config
    root = tmp_path / "checkout with spaces"
    (root / "configs").mkdir(parents=True)
    config = root / "configs/check.json"
    config.write_text(json.dumps({"manifest": "data/input.json", "output": "runs/check"}))
    monkeypatch.chdir(tmp_path)
    loaded = load_config(config)
    assert loaded.manifest == str(root / "data/input.json")
    assert loaded.output == str(root / "runs/check")
