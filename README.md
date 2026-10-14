# FedSER

**Privacy-preserving Speech Emotion Recognition through Semi-Supervised Federated Learning**

Vasileios Tsouvalas · Tanir Ozcelebi · Nirvana Meratnia  
EmotionAware @ IEEE PerCom

Speech emotion recognition depends on sensitive audio that is costly to annotate. This work combines an attention-based speech model with federated self-training, using labeled clips and confident predictions on unlabeled clips while keeping raw audio on each device, then studies label scarcity and speaker distribution in IEMOCAP experiments.

## Project resources

[Project page](https://nousphera.github.io/FedSER/) · [Research overview](https://nousphera.github.io/projects/fedser/) · [A0 poster](https://nousphera.github.io/FedSER/poster.pdf) · [Paper](https://arxiv.org/abs/2202.02611)

This repository hosts the paper’s project page, poster and citation. **The implementation is not included in this release.** Code will be added here after the final paper implementation has been identified and prepared for release.

## Method at a glance

- **Attend to speech features.** A convolutional network processes log-mel spectrograms. Spectral, temporal, and channel attention weight features before the model predicts an emotion.
- **Use unlabeled speech.** Local self-training combines known labels with pseudo-labels from high-confidence predictions. A device-specific confidence schedule accounts for training progress and each client’s participation.
- **Aggregate across devices.** FedAvg combines the locally trained parameters into a shared model. Raw recordings remain on devices, and the method does not require a labeled or unlabeled server dataset.

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

## Materials and licensing

Publication and linked implementation licenses remain with their respective authors and publishers. No license for unreleased code is implied by this resource repository.
