# Paper-to-code mapping

Reference: Tsouvalas, Ozcelebi and Meratnia, *Privacy-preserving Speech Emotion
Recognition through Semi-Supervised Federated Learning*, PerCom Workshops 2022,
[arXiv:2202.02611v1](https://arxiv.org/pdf/2202.02611v1).

This release recovers the author's code and supplies a complete maintained
training path. It has **not reproduced the paper's IEMOCAP scores**. The source
snapshot and the paper disagree in consequential ways. `mode="paper"` follows
the stated method where clear; `mode="source"` exposes recovered behavior for
controlled comparisons. Neither is presented as a bitwise reproduction.

| Component | Maintained code | Paper mode | Source mode / evidence |
|---|---|---|---|
| Four CNN blocks | `fedser/model.py` | Temporal `(4,1)` and spectral `(1,4)` convolutions, concatenation, `1x1` convolution, four-group normalization, ReLU, max pooling, spatial dropout .1; widths 24/32/64/128 | Widths and detailed layer ordering recovered from `src/model/model.py`; L2 .0001 on block convolutions |
| STC attention | `STCAttention` | Mean/max reduction along channels; separate convolutions spanning time and frequency; two linear dense layers with final tanh for channel weights | Original performs spatial 2x2 pooling per channel and uses sigmoid; both modes keep original time-axis softmax, residual addition, global max pooling and classifier because the paper does not fully specify those choices |
| Local objective, Eq. 1 | `method.self_training_loss` | `Ls + beta * Lu`, default beta .5 | `(1-beta)*Ls + beta*Lu`; beta .5 recovered, not specified by paper |
| Pseudo-labels, Eq. 2 | same | Stop-gradient softmax(logits / 2), argmax hard labels, confidence mask | Temperature 1 in recovered client; masked loss divided by **all** unlabeled examples, not accepted count, in both modes |
| Confidence, Eq. 3 | `method.confidence_threshold` | `tau_min + (tau_max-tau_min)/2 * (1-cos(pi*(C-delta*(C-Cs))/R))`; delta .5; C/Cs count completed rounds **before** current fit | Increasing cosine based only on completed client fits; the paper's printed formula decreases and omits the .5 offset. Paper mode is an explicitly documented interpretation of its prose, not a claimed transcription of Eq. 3 |
| FedAvg, Algorithm 1 | `method.aggregate`, Flower strategy | Exact client utterance counts `Nl+Nu`; all model weights aggregated; optimizer recreated per fit | Fixes old client batch-size multiplication/rounded counts; same intended FedAvg weighting |
| Audio | `data.features` | 16 kHz mono, symmetric short padding; 32,000-sample crop in training; nonoverlapping 32,000-sample test segments; incomplete final tail dropped | Original 31,380-sample crop and test stride 6,276; normalize whole waveform before train crop and each test segment separately |
| Log-mel | `data.features` | 400-sample Hann window, 160-sample hop, 1024 FFT, magnitude, 64 mel bins from 60–7800 Hz, natural log after `[1e-5,1e8]` clipping | Exact source DSP values; paper specifies only 25 ms / 10 ms / 64 mel bins |
| IEMOCAP | `data.read_iemocap` | Improvised utterances; neutral/happy/sad/angry; optional explicit excited→happy merge | Later source constants list Neutral/Excited/Sad/Angry. Original benchmark class mapping cannot be recovered unambiguously; strict happy is default and merging is recorded |
| Evaluation protocols | `data.split_records` | Leave one of five sessions out, or five stratified utterance folds | Split utterances before any crops. No segment crosses train/test boundaries |
| Client partitions | `data.partition_clients` | Random quantity variation (default ±25% of mean), or one client per actual training speaker | Seeded bounded integer allocation is a replacement for missing/inconsistent source data dependencies; exact historical partition IDs unavailable |
| Label budgets | `data.label_clients` | Per-client per-class floor of fraction; remainder masked as -1 | Preserves local class proportions subject to rounding. Empty labeled clients fail rather than silently increasing the budget |
| Inference / UA | `train.evaluate` | Average segment probabilities, then macro recall across four emotions; also report ordinary accuracy | Source averaged logits and reported ordinary accuracy. Source mode retains logit averaging, but always reports properly named UA and accuracy |

The source uses a residual `x + x*attention` and global max pooling, while the
paper only describes attention-weighted features and a classifier. These
choices remain shared and qualified rather than inventing a new architecture.
Figure 1 says one-second input; the method text says two seconds, which is the
default here. The same source contains model/data explorations that are not
evidence that the paper used them.

Per-speaker LOSO has eight clients. A stratified overlapping-speaker split of
all five sessions has ten speakers, whereas Table III reports eight devices.
This release does not silently drop two speakers: it derives ownership from
the input manifest and requires the client count to match. Exact reproduction
of that table row requires the original eight-speaker selection, which was not
recovered. No official splits, seeds, checkpoint or training log tying the
snapshot to the published table was found.

Raw audio remains local in a deployed federation. The single-machine simulator
is a research harness. This implementation adds no differential privacy,
secure aggregation, or protection against inference from model updates.
