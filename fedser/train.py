"""One shared local trainer for deterministic simulation and Flower Apps."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import time
import numpy as np
import tensorflow as tf
from .data import features, label_clients, partition_clients, read_wave, split_records
from .method import aggregate, confidence_threshold, self_training_loss, unweighted_accuracy
from .model import build_model


@dataclass(frozen=True)
class RunConfig:
    dataset: str = "synthetic"
    manifest: str = "data/iemocap.json"
    mode: str = "paper"
    protocol: str = "session"
    fold: int = 1
    partition: str = "random"
    clients: int = 2
    quantity_variation: float = .25
    labeled_fraction: float = .1
    supervised_only: bool = False
    rounds: int = 2
    participation: float = .8
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = .001
    beta: float = .5
    temperature: float | None = None
    delta: float = .5
    seed: int = 2021
    synthetic_per_client: int = 40
    output: str = "runs/experiment"

    def __post_init__(self):
        if self.dataset not in {"synthetic", "iemocap"} or self.mode not in {"paper", "source"}:
            raise ValueError("Unknown dataset or method mode")
        if any(not isinstance(x, int) or x < 1 for x in
               (self.clients, self.rounds, self.local_epochs, self.batch_size, self.synthetic_per_client)):
            raise ValueError("Counts must be positive integers")
        if not (0 < self.participation <= 1 and 0 < self.labeled_fraction <= 1
                and 0 <= self.beta <= 1 and 0 <= self.delta <= 1 and self.learning_rate > 0):
            raise ValueError("Invalid training hyperparameters")
        if self.temperature is not None and not self.temperature > 0:
            raise ValueError("temperature must be positive")
        if self.protocol not in {"session", "overlap"} or self.partition not in {"random", "speaker"}:
            raise ValueError("Unknown partition or protocol")
        if self.fold not in range(1, 6) or self.seed < 0 or not 0 <= self.quantity_variation < 1:
            raise ValueError("Invalid fold, seed or quantity variation")

    @property
    def effective_temperature(self):
        return self.temperature if self.temperature is not None else (2. if self.mode == "paper" else 1.)


def load_config(path, **overrides):
    config_path = Path(path).resolve()
    value = json.loads(config_path.read_text())
    # Shipped configs live under configs/; keep their data paths tied to the
    # checkout even when Flower executes a bundled app from a different cwd.
    base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    for key in ("manifest", "pretrained_encoder", "output"):
        if value.get(key) and not Path(value[key]).is_absolute():
            value[key] = str((base / value[key]).resolve())
    value.update({k: v for k, v in overrides.items() if v is not None})
    return RunConfig(**value)


def make_data(config):
    """Prepare ownership once; labels -1 are never read as ground truth."""
    if config.dataset == "synthetic":
        rng = np.random.default_rng(config.seed)
        def sample(cid, n, test=False):
            label = n % 4
            samples = 32000 * (2 if test else 1)
            t = np.arange(samples) / 16000
            wave = (.2 * np.sin(2 * np.pi * (160 + 130 * label) * t)
                    + rng.normal(0, .02, samples)).astype(np.float32)
            return {"id": f"{'test' if test else 'train'}-{cid}-{n}", "label": label,
                    "speaker": f"synthetic-{cid}", "_wave": wave}
        clients = [[sample(cid, n) for n in range(config.synthetic_per_client)] for cid in range(config.clients)]
        test = [sample(99, n, True) for n in range(8)]
    else:
        manifest = json.loads(Path(config.manifest).read_text())
        from .preparation import check_records
        root = Path(manifest["root"])
        check_records(manifest["records"], list(range(4)), None, root)
        if any(type(r.get("session")) is not int or r["session"] not in range(1, 6)
               or not isinstance(r.get("speaker"), str) or not r["speaker"] for r in manifest["records"]):
            raise ValueError("Invalid IEMOCAP session/speaker metadata")
        rows = [{**r, "path": str(root / r["path"])} for r in manifest["records"]]
        train, test = split_records(rows, config.protocol, config.fold, config.seed)
        clients = partition_clients(train, config.partition, config.clients, config.quantity_variation, config.seed)
    clients = label_clients(clients, config.labeled_fraction, config.seed)
    if config.supervised_only:
        clients = [[r for r in client if r["label"] >= 0] for client in clients]
    return clients, test, (198 if config.mode == "paper" else 194, 64, 1)


def waveform(record):
    return record["_wave"] if "_wave" in record else read_wave(record["path"])


def local_train(weights, records, shape, config, completed, client_completed, client_id):
    """One fit creates a fresh Adam optimizer as in the recovered client."""
    tf.keras.backend.clear_session()
    seed = int(np.random.SeedSequence([config.seed, completed, client_id]).generate_state(1)[0])
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(shape, mode=config.mode)
    model.set_weights(weights)
    optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    threshold = confidence_threshold(completed, client_completed, config.rounds, delta=config.delta, mode=config.mode)
    labeled = [r for r in records if r["label"] >= 0]
    unlabeled = [r for r in records if r["label"] == -1]
    if not labeled:
        raise ValueError("Local self-training requires at least one labeled utterance")
    total_loss, accepted, num_unlabeled, steps = 0., 0, 0, 0
    for _ in range(config.local_epochs):
        order = rng.permutation(len(unlabeled) if unlabeled else len(labeled))
        label_order = rng.permutation(len(labeled))
        for offset in range(0, len(order), config.batch_size):
            indices = order[offset:offset + config.batch_size]
            if unlabeled:
                # Equally sized labeled/unlabeled batches; cycle the labeled set.
                lindices = label_order[np.arange(offset, offset + len(indices)) % len(labeled)]
                batch = [labeled[i] for i in lindices] + [unlabeled[i] for i in indices]
            else:
                batch = [labeled[i] for i in indices]
            x = np.stack([features(waveform(r), True, rng, mode=config.mode) for r in batch])
            y = np.array([r["label"] for r in batch], np.int32)
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss, details = self_training_loss(logits, y, threshold, config.effective_temperature,
                                                  config.beta, mode=config.mode)
                loss += tf.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            tf.debugging.assert_all_finite(loss, "Nonfinite local loss")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += float(loss)
            accepted += int(details["accepted"])
            num_unlabeled += int(details["unlabeled"])
            steps += 1
    return model.get_weights(), len(records), {"loss": total_loss / steps, "threshold": threshold,
            "accepted": accepted, "unlabeled": num_unlabeled, "steps": steps}


def evaluate(model, records, mode="paper"):
    labels, predictions, losses = [], [], []
    for row in records:
        x = features(waveform(row), False, np.random.default_rng(0), mode=mode)
        logits = model(x, training=False)
        if mode == "paper":
            scores = tf.reduce_mean(tf.nn.softmax(logits), axis=0).numpy()
        else:
            scores = tf.nn.softmax(tf.reduce_mean(logits, axis=0)).numpy()
        labels.append(row["label"])
        predictions.append(int(scores.argmax()))
        losses.append(float(-np.log(max(float(scores[row["label"]]), 1e-12))))
    return {"ua": unweighted_accuracy(labels, predictions),
            "accuracy": float(np.mean(np.asarray(labels) == predictions)), "eval_loss": float(np.mean(losses))}


def prepare_output(config, clients, test, engine):
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    value = asdict(config)
    value["effective_temperature"] = config.effective_temperature
    (output / "config.json").write_text(json.dumps(value, indent=2) + "\n")
    public_rows = [[{k: v for k, v in r.items() if k != "_wave"} for r in c] for c in clients]
    (output / "partitions.json").write_text(json.dumps({"clients": public_rows, "test_ids": [r["id"] for r in test]}, indent=2) + "\n")
    environment = {"python": platform.python_version(), "platform": platform.platform(), "engine": engine,
                   "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                   "devices": [d.name for d in tf.config.list_physical_devices()]}
    if config.dataset == "iemocap":
        environment["manifest_sha256"] = hashlib.sha256(Path(config.manifest).read_bytes()).hexdigest()
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    return output


def run(config):
    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()
    clients, test, shape = make_data(config)
    output = prepare_output(config, clients, test, "sequential-fedavg")
    model = build_model(shape, mode=config.mode)
    weights = model.get_weights()
    rng = np.random.default_rng(config.seed)
    participation = [0] * config.clients
    history = []
    start = time.monotonic()
    for round_index in range(config.rounds):
        selected = sorted(rng.choice(config.clients, max(1, int(config.clients * config.participation)), replace=False).tolist())
        updates, training = [], []
        for cid in selected:
            update, count, metrics = local_train(weights, clients[cid], shape, config, round_index, participation[cid], cid)
            updates.append((update, count))
            training.append({"client": cid, "num_examples": count, **metrics})
            participation[cid] += 1
        weights = aggregate(updates)
        model.set_weights(weights)
        result = {"round": round_index + 1, "selected_clients": selected,
                  "client_metrics": training, **evaluate(model, test, config.mode)}
        history.append(result)
        print(json.dumps(result), flush=True)
    model.save(output / "model.keras")
    result = {"benchmark_reproduction": False, "dataset": config.dataset, "mode": config.mode,
              "seconds": time.monotonic() - start, "rounds": history}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
