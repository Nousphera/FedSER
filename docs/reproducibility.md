# Experiment guide

## Environment

Use Python 3.11 or 3.12 and the [setup script](../scripts/setup_env.sh).
`requirements-lock-py311.txt` records a Python 3.11.15/macOS arm64 CPU environment
with TensorFlow 2.20.0, Keras 3.15.1, Flower 1.40.0 and Ray 2.55.1.
Install `.[flower,test]` for Flower and tests or `.[test]` for sequential runs.
Run `python -m pip check` and `python -m pytest -q` after installation.

The test suite covers loss/gradient definitions, thresholds, exact aggregation,
prepared WAV data, full-model updates, Flower message exchange and Keras
serialization. Importing the package does not download data or start training.

## Data and splits

`fedser prepare` reads licensed IEMOCAP annotations and writes a manifest with
`schema`, `root`, `merge_excited` and `records`. Each record includes an utterance
ID, relative WAV path, session, speaker, integer label, SHA-256 and byte count.
The speaker comes from the utterance's F/M suffix. Audio must be 16 kHz PCM;
stereo inputs use their first channel. Preparation decodes every file and
rejects duplicate IDs/paths. `fedser validate --manifest PATH` repeats decoding
and hash checks. Run it after copying or changing a corpus.

Labels are 0 neutral, 1 happy, 2 sad and 3 angry. Excited-to-happy merging requires
`--merge-excited`. Preserve this policy and the manifest hash with each run.
Manifests and run records can contain local paths; share them according to your
dataset agreement.

`protocol="session"`, `fold=1..5` holds out a complete session. `protocol="overlap"`
uses five stratified utterance folds. Train/test utterances are split before
cropping. Held-out data never enters local training or pseudo-label selection.

`partition="random"` creates bounded integer client sizes around the mean
(default ±25%; integer rounding can extend a bound by one example).
`partition="speaker"` assigns each training speaker to one client and requires
the configured count to match. Full-corpus session holdout has eight training
speakers; overlapping folds normally contain ten.

Per-client/class label budgets use `floor(class_count*labeled_fraction)`.
Remaining labels are masked as `-1`, so small clients can have a lower realized
fraction. Empty labeled clients fail. `supervised_only=true` uses exactly the
same label masks and then removes unlabeled examples. `labeled_fraction=1.0`
provides the fully labeled comparison.

## Configuration and repeated runs

The IEMOCAP config uses 100 rounds, 80% participation, one local epoch, batch
size 64 **per labeled or unlabeled branch**, Adam 0.001, beta 0.5, temperature 2,
and confidence 0.5–0.9 with delta 0.5. The [method guide](method.md) defines mode
and epoch semantics.

Use a distinct output directory for every fold/seed. For example, sequential
experiments can be scheduled with:

```bash
for fold in 1 2 3 4 5; do
  for seed in 2021 2022 2023 2024 2025; do
    fedser train --config configs/iemocap.json --fold "$fold" --seed "$seed" \
      --output "runs/iemocap-fold${fold}-seed${seed}"
  done
done
```

These example seeds control folds, allocations, label masks, model initialization,
client sampling and crops. Report unweighted accuracy across runs within each
split, then the split average and variation. Keep ordinary accuracy separately
named. The sequential CPU engine enables deterministic TensorFlow operations;
hardware/library changes and Flower runtime node-ID assignment can change
trajectories. Preserve participation records for comparisons.

## Outputs

| File | Contents |
|---|---|
| `config.json` | Resolved parameters and effective temperature |
| `partitions.json` | Client utterance ownership, visible/masked labels and held-out IDs |
| `environment.json` | Python/platform/packages/devices, runtime engine and manifest hash |
| `metrics.json` | Dataset, mode, elapsed time, round evaluation and client fit metrics |
| `model.keras` | Final attention model; load after `import fedser.model` |

`ua` is mean per-class recall; `accuracy` is ordinary utterance accuracy, both
fractions in `[0,1]`. `eval_loss` is mean true-label negative log-probability.
`accepted` and `unlabeled` count batch exposures; `num_examples`/`num-examples`
is the distinct utterance count used by FedAvg. Flower also records evaluation
at round 0. Output directories are created once. Training runs for the configured
rounds, without automatic resume, early stopping or test-set model selection.

## Native Flower CLI

The README's `flwr run . --stream` starts the packaged ClientApp and ServerApp
through Flower 1.40's local SuperLink. Set `num-supernodes` to the configured
client count. Use absolute `config` and `output` paths in `--run-config`.

Create the virtual environment in a path without spaces, including the actual
package location; a symlink to a space-containing environment is insufficient
for Ray workers. Repository and dataset paths may contain spaces. Paths inside
files in `configs/` resolve against the checkout. Other config files resolve
paths against their own directory; CLI output overrides use the caller's
working directory.

`FLWR_HOME`, `FLWR_LOCAL_SUPERLINK_HTTP_API_PORT` and
`FLWR_LOCAL_CONTROL_API_PORT` isolate the app's local runtime. The README sets
`FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` to use the setup environment.
Flower keeps its SuperLink service running between jobs. Use `flwr list`,
`flwr log RUN_ID --show` and `flwr stop RUN_ID` with the same environment.
Stopping a run leaves the service available for future runs. Consult the
[local runtime guide](https://flower.ai/docs/framework/main/en/how-to-run-flower-locally.html)
for service lifecycle and logs; when switching environments, stop only the
service associated with that `FLWR_HOME`.

Confirm the run reports completion and inspect the saved metrics/model files.
`flwr build` packages the app. The Python `flower` subcommand runs the same apps
through Ray, and `train` runs the shared trainer sequentially for debugging.
Deployment across machines additionally requires client-local data provisioning,
authenticated transport and a server-side evaluation policy.
