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
this choice fixed when comparing runs; source constants also include an
excited-based class selection.

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
are separate deployment requirements. See the [experiment guide](reproducibility.md)
for output records and runtime management, and [source credits](../provenance/README.md)
for attribution.
