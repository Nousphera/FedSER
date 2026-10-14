# FedSER

**Privacy-preserving Speech Emotion Recognition through Semi-Supervised Federated Learning**
Vasileios Tsouvalas, Tanir Ozcelebi and Nirvana Meratnia · EmotionAware / PerCom Workshops 2022

[Paper](https://arxiv.org/abs/2202.02611) · [Project](https://nousphera.github.io/FedSER/) · [Poster](https://nousphera.github.io/FedSER/poster.pdf) · [Method mapping](docs/method.md) · [Reproducibility](docs/reproducibility.md)

FedSER trains an attention CNN on local labeled and unlabeled speech, selects
confident pseudo-labels, and combines client models with FedAvg. This release
contains the recovered author source, a maintained TensorFlow implementation,
an IEMOCAP adapter, deterministic experiments and a runnable Flower App.

**Status:** installation, 17 tests, file-based audio training and two-round real
Flower simulation verified. Full IEMOCAP benchmark reproduction remains
unverified. The recovered source differs from the paper; the default `paper`
mode follows the stated method with explicitly documented interpretations.
`source` mode supports comparisons with the recovered behavior.

## Install and run

Python **3.11 or 3.12**. A CPU is enough for the smoke test; use an environment
path without spaces for Ray/Flower. From this repository:

```bash
python3.11 -m venv ~/.venvs/fedser
source ~/.venvs/fedser/bin/activate
python -m pip install -e '.[flower,test]' -c requirements-lock-py311.txt
python -m pip check
python -m pytest -q

# Small deterministic FedAvg run; no data download
fedser train --config configs/smoke.json

# Actual Flower ClientApp + ServerApp, using the same trainer
fedser flower --config configs/smoke.json --output runs/flower-smoke
```

The smoke test uses generated waveforms and the full four-block STC model.
It verifies training and aggregation, not emotion-recognition performance.
An existing output directory is never overwritten. The native Flower App
entry points are declared in `pyproject.toml`; `flwr build` packages the app.

## Train on IEMOCAP

Obtain IEMOCAP under its [dataset access terms](https://sail.usc.edu/iemocap/).
Keep the original `Session*/dialog/EmoEvaluation` annotations and
`Session*/sentences/wav` utterances. No recordings are included or downloaded.

```bash
fedser prepare --root /path/to/IEMOCAP_full_release --output data/iemocap.json
fedser train --config configs/iemocap.json
# Or use the Flower runtime:
fedser flower --config configs/iemocap.json --output runs/iemocap-flower
```

Preparation keeps improvised neutral/happy/sad/angry utterances. Excited speech
is excluded by default; `prepare --merge-excited` explicitly merges it into
happy and records that choice. The paper and recovered data helpers do not
unambiguously establish the original mapping.

Edit the JSON configuration to choose session-held-out or overlapping-speaker
folds, random or per-speaker clients, label fraction, supervised-only baseline,
rounds and local training. Session-held-out per-speaker training has eight
clients. See [reproducibility](docs/reproducibility.md) for all folds/seeds,
label-budget comparisons and the unresolved eight-client overlapping protocol.

Each run saves its resolved configuration, ownership/label partitions,
environment, per-round client metrics, utterance-level UA and accuracy, and a
loadable `model.keras`. The data split is made at utterance level before
cropping. The test set is used only for evaluation.

## Research boundaries and provenance

The [method mapping](docs/method.md) documents the threshold-equation mismatch,
attention, loss weighting, temperature, segmentation and metric corrections.
[Exact source files and hashes](provenance/README.md) identify the recovered
commit without publishing private history, data, checkpoints or result archives.
No paper checkpoint or original benchmark split manifest was recovered.

The supplied runtimes are single-machine research simulations with local file
access. Production data isolation, transport authentication and deployment
hardening are outside this release. Keeping recordings out of model messages
provides no formal differential-privacy or secure-aggregation guarantee.

## Citation

```bibtex
@inproceedings{tsouvalas2022fedser,
  author = {Tsouvalas, Vasileios and Ozcelebi, Tanir and Meratnia, Nirvana},
  title = {Privacy-preserving Speech Emotion Recognition through Semi-Supervised Federated Learning},
  booktitle = {2022 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops)},
  year = {2022},
  pages = {359--364},
  note = {EmotionAware workshop},
  doi = {10.1109/PerComWorkshops53856.2022.9767445},
  url = {https://doi.org/10.1109/PerComWorkshops53856.2022.9767445}
}
```

## Licensing

The author authorized publication of the recovered code. The original
repository had no code license; this release does not invent reuse terms.
Contact the author for permission until an explicit license is provided.
IEMOCAP, publications and dependencies retain their respective licenses.
See [provenance](provenance/README.md) for exclusions and attribution.
