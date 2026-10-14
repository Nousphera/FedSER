# Reproducibility

## Environment and validation

Validated with Python 3.11.15 on macOS arm64 CPU, TensorFlow 2.20.0, Keras 3.15.1,
Flower 1.40.0 and Ray 2.55.1. `requirements-lock-py311.txt` records the complete
resolved dependency set. Python 3.12 is within dependency support but was not
independently tested. Linux/CUDA was not tested in this release.

The core package has no Flower dependency unless the `flower` extra is chosen.
Install with `python -m pip install -e '.[test]'` for sequential experiments.
No dataset downloads or training occur during import, build or test collection.

The release validation includes 17 tests, two-round sequential and real Flower
simulations with the full attention model, an IEMOCAP-shaped annotation/WAV
fixture through actual decoding and training, and Keras save/load equivalence.
These are integration checks; **no IEMOCAP benchmark result is claimed**.

Ray 2.55.1 failed to launch workers when the Python environment path contained
spaces on the tested macOS system. Install the virtual environment at a path
without spaces (the README example does this). The repository itself can remain
elsewhere. The successful Flower smoke used an isolated environment path without
spaces and exchanged actual Flower Message API train/reply messages.

## Dataset and split contract

`fedser prepare` reads the original annotations and writes a JSON manifest:
`schema`, `root`, `merge_excited`, and `records`. Each record has `id`, relative
WAV `path`, integer `session`, actual utterance `speaker`, and integer `label`.
The speaker comes from the utterance suffix (F/M), not the dialogue-name prefix.
WAV files must be mono-compatible 16 kHz PCM; a rate mismatch raises an error.
Stereo WAVs decode their first channel. Preparation validates paths and duplicate
utterance IDs. Audio is decoded and checked during training/evaluation.

Labels are 0 neutral, 1 happy, 2 sad, 3 angry. Excited-to-happy merging requires
an explicit preparation flag. Preserve the manifest and its recorded SHA-256
with each experiment; do not upload licensed files or manifests containing
private paths to a public repository.

`protocol="session"`, `fold=1..5` holds out that session's dyad. `overlap` uses
five stratified utterance folds. A seed determines folds, client allocation,
class-stratified label masks, model initialization, client sampling and crops.
Train/test utterances are disjoint before preprocessing. Test data never enters
local training, confidence selection, client aggregation or model selection.

`partition="random"` creates the requested client count with bounded integer
sizes around the mean (default ±25%; rounding can extend bounds by one sample).
`partition="speaker"` assigns each training speaker to exactly one client and
requires the count to match. Full-corpus LOSO has eight training speakers;
full-corpus overlapping folds normally have ten. The paper's eight-device
overlapping-speaker row needs an original speaker selection that was not
recovered; do not silently discard two speakers to force that count.

Label allocation keeps `floor(class_count * labeled_fraction)` examples per
client/class, then masks all remaining labels to -1. The realized fraction can
be lower than the requested fraction, especially on small clients; exact masks
are saved. Empty labeled clients raise an error. `supervised_only=true` removes
unlabeled examples after constructing exactly the same labeled masks, allowing
an equal-label baseline. Set `labeled_fraction=1.0` for the fully labeled
reference. No paper table can be reproduced merely by changing these switches
without resolving the documented original-source ambiguities.

## Run the full protocol

The default IEMOCAP configuration uses 100 rounds, 80% client participation,
one local epoch, batch size 64 **per labeled or unlabeled branch**, Adam .001,
beta .5, temperature 2, and confidence .5–.9 with delta .5. One local epoch
traverses the unlabeled subset while cycling the labeled subset. For a
supervised-only client it traverses its labeled subset. The source's meaning of
one local “step” is a local epoch, not one optimizer update.

Run from the checkout with a new output directory for every fold/seed:

```bash
for fold in 1 2 3 4 5; do
  for seed in 2021 2022 2023 2024 2025; do
    fedser train --config configs/iemocap.json --fold "$fold" --seed "$seed" \
      --output "runs/iemocap-fold${fold}-seed${seed}"
  done
done
```

Those are release example seeds; the exact historical five benchmark seeds
were not recovered. Aggregate UA over all runs within each split, then report
the split average and variation. Do not compare single-run weighted accuracy
with the paper's averaged unweighted accuracy.

The sequential engine is deterministic on the tested CPU. Cross-device/library
bitwise equality is not guaranteed. Flower samples sorted runtime node IDs
with a seed; different runtime node-ID assignments can change which logical
client is selected under partial participation. Every reply's node/client ID
and exact sample count is saved, so preserve those logs when comparing runs.
The full-participation smoke avoids that assignment ambiguity.

## Output schema

| File | Contents |
|---|---|
| `config.json` | All resolved settings, including effective temperature |
| `partitions.json` | Per-client utterance metadata and visible/-1 labels; held-out utterance IDs |
| `environment.json` | Python/platform/packages/devices, runtime engine, input manifest SHA-256 |
| `metrics.json` | Dataset/mode, elapsed seconds, `benchmark_reproduction=false`, round metrics and per-client fit details |
| `model.keras` | Final attention model, loadable after `import fedser.model` |

`ua` is mean per-class recall and `accuracy` is ordinary utterance accuracy,
both fractions in [0,1]. `eval_loss` is mean negative log-probability of the true
utterance label. `accepted` and `unlabeled` count batch exposures across local
epochs, while `num_examples`/`num-examples` is the **distinct client utterance
count used for FedAvg**. Flower records an additional initial evaluation at
round 0. Training artifacts are local outputs, not supplied paper checkpoints.

There is no automatic checkpoint resume, early stopping or test-set tuning.
Existing output directories are rejected to preserve experiment records.

## Flower entry points

`fedser flower --config ...` runs the packaged ClientApp and ServerApp through
Flower's Ray backend. Both use the same local trainer, loss and model as the
sequential engine. Client participation survives actor recreation in
`Context.state`, and a failed selected client aborts aggregation rather than
silently changing the selected cohort. The server uses held-out data solely
for research evaluation; the optimization requires no server training data.

`flwr build` validates and packages the native App described in `pyproject.toml`.
Deployment requires separate provisioning of local client manifests/data,
authenticated transport and a chosen server-side evaluation policy. The
provided shared-filesystem simulation is not a production privacy boundary.

API references: [Flower TensorFlow App](https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html),
[Keras GroupNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GroupNormalization).


## Native Flower CLI

The README's `flwr run . --stream` runs the packaged ClientApp and ServerApp
through Flower 1.40's managed local SuperLink. Both native CLI paths were verified
with two clients and two rounds; client metrics, server metrics and model files
were saved. Generated-data execution is not a dataset benchmark.

`FLWR_HOME` and the local HTTP port in the README isolate this app's runtime from
other projects. Flower keeps the managed SuperLink running after a run. Use
`flwr list`, `flwr log RUN_ID --show`, or `flwr stop RUN_ID` in the same environment.
A stop request stops that run, not the background service. The official
[local runtime guide](https://flower.ai/docs/framework/main/en/how-to-run-flower-locally.html)
explains service lifecycle and logs. If changing virtual environments, inspect
and stop only the matching service process before starting its replacement.
`FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` uses the dependencies already
installed by setup; this avoids a second uncontrolled installation at run time.

The Python environment and its actual package files must live in paths without
spaces; symlinking a space-containing environment is insufficient. The repository
and data paths may contain spaces. Supply absolute `config` and `output` paths in
`--run-config`, as shown in the README. Paths inside files in `configs/` resolve
against that directory's parent (the checkout), even when Flower executes its
bundle elsewhere. For a config outside `configs/`, paths resolve against the
config file's own directory. CLI `--output` overrides resolve in the caller's
working directory. Set `num-supernodes` equal to the configured client count.

Preparation now decodes every input and records SHA-256/byte counts before
atomically publishing its manifest. `fedser validate --manifest PATH` checks those
hashes again. Training performs structural, label and duplicate-path checks;
run explicit validation after moving or changing data. Preparation and training
refuse to overwrite outputs. Manifests are local artifacts and can reveal paths.
