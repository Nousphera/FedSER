# FedSER

**Privacy-preserving Speech Emotion Recognition through Semi-Supervised Federated Learning**

Vasileios Tsouvalas · Tanir Ozcelebi · Nirvana Meratnia · EmotionAware @ IEEE PerCom 2022

[Paper](https://ieeexplore.ieee.org/document/9767445) · [Project](https://nousphera.github.io/FedSER/) · [Poster](https://nousphera.github.io/FedSER/poster.pdf)

## Abstract

Speech Emotion Recognition (SER) refers to the recognition of human emotions from natural speech. If done accurately, it can offer a number of benefits in building human-centered context-aware intelligent systems. Existing SER approaches are largely centralized, without considering users' privacy. Federated Learning (FL) is a distributed machine learning paradigm dealing with decentralization of privacy-sensitive personal data. In this paper, we present a privacy-preserving and data-efficient SER approach by utilizing the concept of FL. To the best of our knowledge, this is the first federated SER approach, which utilizes self-training learning in conjunction with federated learning to exploit both labeled and unlabeled on-device data. Our experimental evaluations on the IEMOCAP dataset shows that our federated approach can learn generalizable SER models even under low availability of data labels and highly non-i.i.d. distributions. We show that our approach with as few as 10% labeled data, on average, can improve the recognition rate by 8.67% compared to the fully-supervised federated counterparts.

## Setup

Python 3.11 is tested on macOS arm64 CPU; Python 3.12 is supported but untested.
A CPU is sufficient for the smoke run. Put the environment at a path without
spaces so Flower/Ray can start workers.

```bash
git clone https://github.com/Nousphera/FedSER.git
cd FedSER
bash scripts/setup_env.sh "$HOME/.venvs/fedser"
source "$HOME/.venvs/fedser/bin/activate"
python -m pytest -q
```

The setup script creates a new virtual environment and installs the package,
Flower simulation and test dependencies using the tested dependency constraints.
It refuses to change an existing environment. The equivalent manual install is
`python3.11 -m venv "$HOME/.venvs/fedser"`, activate it, then
`python -m pip install -e '.[flower,test]' -c requirements-lock-py311.txt`.

## Dataset

Request [IEMOCAP](https://sail.usc.edu/iemocap/) from its provider and extract the
licensed release yourself. Preserve `Session*/dialog/EmoEvaluation/*.txt` and
`Session*/sentences/wav/<dialogue>/*.wav`. No access credentials or audio are supplied.

```bash
python scripts/prepare_data.py prepare --root /path/to/IEMOCAP_full_release \
  --output data/iemocap.json
python scripts/prepare_data.py validate --manifest data/iemocap.json
```

Preparation indexes improvised neutral/happy/sad/angry utterances, decodes every
16 kHz PCM WAV, and atomically writes IDs, speaker/session labels and SHA-256
checksums. Existing manifests are preserved. To merge excited into happy, add
`--merge-excited`; this choice is recorded. The original class mapping remains
ambiguous; see [data and split details](docs/reproducibility.md).

## Run

Start a native Flower App from the repository. The smoke configuration trains the
full attention CNN with two clients for two rounds on generated waveforms:

```bash
export FLWR_HOME="$HOME/.cache/fedser-flower"
export FLWR_LOCAL_SUPERLINK_HTTP_API_PORT=39131
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
flwr run . --stream \
  --run-config "config='$(pwd)/configs/smoke.json' output='$(pwd)/runs/flower-smoke'" \
  --federation-config 'num-supernodes=2 client-resources-num-cpus=1 init-args-num-cpus=2'
```

After preparing IEMOCAP, run the ten-client example configuration:

```bash
flwr run . --stream \
  --run-config "config='$(pwd)/configs/iemocap.json' output='$(pwd)/runs/iemocap-flower'" \
  --federation-config 'num-supernodes=10 client-resources-num-cpus=1 init-args-num-cpus=2'
```

The federation node count must equal `clients` in the JSON config. The server
saves evaluation results and the final model; each client performs actual local
self-training. For sequential debugging, use
`fedser train --config configs/smoke.json --output runs/sequential-smoke`.
Flower keeps its local SuperLink running; see [runtime lifecycle and path rules](docs/reproducibility.md#native-flower-cli).

## Outputs and reproducibility

Each new output directory contains `config.json`, `partitions.json`,
`environment.json`, `metrics.json`, and `model.keras`. Partitions save utterance
ownership and label masks; metrics include utterance accuracy, unweighted accuracy
and client participation. Choose a new output directory for each run.

The [method mapping](docs/method.md), [reproducibility guide](docs/reproducibility.md)
and [source provenance](provenance/README.md) distinguish the paper interpretation
from recovered behavior. Full IEMOCAP benchmarks, GPU execution and original
checkpoint equivalence remain unverified. This local research simulation adds no
formal privacy guarantee.

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

The recovered source has no code license; reuse terms remain unspecified.
Contact the authors for permission. Dataset, publication and dependency terms
remain with their providers; see [source provenance](provenance/README.md).
