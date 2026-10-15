# Method and configuration

FedSER combines an attention CNN with federated self-training for four-class
speech emotion recognition. The method is described in
[Tsouvalas, Ozcelebi and Meratnia, PerCom Workshops 2022](https://arxiv.org/pdf/2202.02611v1).

## Model and audio

`fedser/model.py` implements four convolutional blocks with widths 24, 32, 64 and
128. Each block concatenates temporal `(4,1)` and spectral `(1,4)` convolutions,
then applies a `1x1` convolution, four-group normalization, ReLU, max pooling and
spatial dropout 0.1. Block convolutions use L2 regularization 0.0001.
Spectral, temporal and channel attention is followed by residual addition
`x + x*attention`, global max pooling and a four-class classifier.

Audio uses 16 kHz mono-compatible PCM. Log-mel extraction uses a 400-sample Hann
window, 160-sample hop, 1024-point FFT, magnitude, 64 mel bins from 60–7800 Hz,
and natural log after clipping to `[1e-5,1e8]`. Short clips are symmetrically padded.
Training normalizes the waveform before cropping; evaluation normalizes each
segment. The default two-second crop follows the paper's method text.

## Training modes

Set `mode` in the JSON configuration. The default `paper` mode uses the paper's
objective and two-second inputs; `source` provides the alternative formulas and
preprocessing of the author's source implementation.

| Setting | `paper` | `source` |
|---|---|---|
| Channel attention | Mean/max channel reduction and two linear dense layers, final tanh | Spatial 2×2 pooling per channel, sigmoid |
| Attention shared by both | Time-axis softmax, residual addition, global max pooling | Same |
| Local loss | `Ls + beta * Lu` | `(1-beta) * Ls + beta * Lu` |
| Default `beta` | 0.5 | 0.5 |
| Pseudo-label temperature | 2 | 1 |
| Training crop | 32,000 samples | 31,380 samples |
| Evaluation segments | Nonoverlapping 32,000-sample windows | 31,380-sample windows with stride 6,276 |
| Segment aggregation | Mean class probabilities | Mean logits |
| Confidence schedule | Global completed rounds with client participation adjustment | Completed local fits |

In both modes, pseudo-labels are detached hard argmax labels. A confidence mask
selects pseudo-labels for the unlabeled loss, whose denominator includes **all**
unlabeled examples. One local epoch traverses the unlabeled subset while cycling
the labeled subset. Supervised-only clients traverse their labeled subset.
Each fit creates a fresh Adam optimizer. FedAvg weights updates by the exact
number of distinct client utterances, `Nl + Nu`, and aggregates all model weights.

The default confidence schedule is

```text
tau_min + (tau_max-tau_min)/2 * (1-cos(pi*(C-delta*(C-Cs))/R))
```

Here `C` and `Cs` count global and client-completed rounds before the fit, `R` is
the total round count, and `delta=0.5`. This increasing schedule follows the
paper's prose; its printed Eq. 3 has a different sign/offset. The formula above
is the explicit implementation contract.

## Labels, splits and evaluation

IEMOCAP preparation selects improvised neutral, happy, sad and angry utterances.
`--merge-excited` explicitly includes excited speech in the happy class. Keep
this choice fixed when comparing runs.

Session evaluation holds out one complete session before preprocessing. The
alternative `overlap` protocol uses five stratified utterance folds. Random
partitions use seeded bounded integer allocation, with default quantity
variation ±25% of the mean. Speaker partitions use one client per actual
training speaker. Full-corpus session holdout gives eight clients; overlapping
folds normally include all ten speakers. An eight-client overlapping experiment
therefore requires an explicitly selected eight-speaker input population.

Each client/class receives `floor(class_count * labeled_fraction)` labels; other
labels are masked as `-1`. Empty labeled clients fail with an actionable error.
Utterance predictions aggregate segments before computing ordinary accuracy
and unweighted accuracy (mean recall over the four emotions).

## Federation

The sequential engine and native Flower App share the same model, local trainer
and aggregation rules. A selected client failure aborts aggregation. Flower
persists client participation state in `Context.state` across actor recreation.
Its seeded selection uses runtime node IDs, so partial-participation trajectories
can differ across launches or from sequential execution.

Training exchanges model updates; differential privacy and secure aggregation
are separate deployment requirements. Execution settings and output records are defined below.

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

`protocol="session"` uses `fold=1..5`; `protocol="overlap"` uses five stratified
utterance folds. The client and label-budget rules are defined above.
`supervised_only=true` uses the same label masks and removes unlabeled examples;
`labeled_fraction=1.0` selects the fully labeled comparison.

## Configuration and repeated runs

The IEMOCAP config uses 100 rounds, 80% participation, one local epoch, batch
size 64 **per labeled or unlabeled branch**, Adam 0.001, beta 0.5, temperature 2,
and confidence 0.5–0.9 with delta 0.5. Mode and epoch semantics are defined above.

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

Run the README's `flwr run . --stream` command and match `num-supernodes` to the
configured client count. Pass absolute `config` and `output` paths. The virtual
environment's real path must not contain spaces; repository and data paths may.
Files in `configs/` resolve paths against the checkout; other config files resolve
paths against their own directory. CLI output overrides use the working directory.

Use the same `FLWR_HOME` for `flwr list`, `flwr log RUN_ID --show` and
`flwr stop RUN_ID`. Stopping a run leaves its local SuperLink available. When
changing environments, stop only that environment's service; see the
[Flower runtime guide](https://flower.ai/docs/framework/main/en/how-to-run-flower-locally.html).
The README sets isolated ports and disables automatic runtime dependency installation.

Check the completed run status and saved metrics/model files. `flwr build`
packages the app; the Python `flower` command uses the same apps through Ray,
and `train` executes the shared trainer sequentially. Deployment across machines
requires client-local data, authenticated transport and an evaluation policy.
