"""FedSER's four-block CNN with spectral, temporal and channel attention."""

import torch
from torch import nn


def _initialize(module):
    # Match the Keras Conv2D/Dense Glorot-uniform and zero-bias defaults.
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class AudioBlock(nn.Module):
    """Separate spectral/temporal filters followed by a joint projection."""

    def __init__(self, in_channels, width, dropout):
        super().__init__()
        self.spectral = nn.Sequential(
            # SAME padding for an even kernel puts the extra sample last.
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(in_channels, width, (1, 4)),
            nn.GroupNorm(4, width, eps=1e-3),
            nn.ReLU(),
        )
        self.temporal = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 2)),
            nn.Conv2d(in_channels, width, (4, 1)),
            nn.GroupNorm(4, width, eps=1e-3),
            nn.ReLU(),
        )
        self.joint = nn.Sequential(
            nn.Conv2d(2 * width, width, 1),
            nn.GroupNorm(4, width, eps=1e-3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.joint(torch.cat((self.spectral(x), self.temporal(x)), dim=1))


class STCAttention(nn.Module):
    """Channel-pooled ST maps and a tanh channel map, normalized over time."""

    def __init__(self):
        super().__init__()
        # Four blocks map [1,198,64] inputs to [128,12,4]. The spectral
        # filter collapses time, and the temporal filter collapses frequency.
        self.spectral = nn.Conv2d(2, 1, (12, 1))
        self.temporal = nn.Conv2d(2, 1, (1, 4))
        self.channel = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 128),
            nn.Tanh(),
        )
        self.apply(_initialize)

    def forward(self, x):
        pooled = torch.cat((x.amax(dim=1, keepdim=True), x.mean(dim=1, keepdim=True)), dim=1)
        spectral_temporal = self.spectral(pooled) * self.temporal(pooled)
        channel = self.channel(x.mean(dim=(-2, -1)))[:, :, None, None]
        return (spectral_temporal * channel).softmax(dim=2)


class FedSERClassifier(nn.Module):
    """Four-emotion classifier returning logits for [batch,1,198,64] inputs."""

    def __init__(self, dropout=0.1):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        channels = (1, 24, 32, 64, 128)
        self.blocks = nn.Sequential(
            *(AudioBlock(a, b, dropout) for a, b in zip(channels[:-1], channels[1:]))
        )
        self.attention = STCAttention()
        self.classifier = nn.Linear(128, 4)
        self.apply(_initialize)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 198, 64):
            raise ValueError("Expected spectrograms with shape [batch,1,198,64]")
        x = self.blocks(x)
        x = x + x * self.attention(x)
        return self.classifier(x.amax(dim=(-2, -1)))


def build_model(dropout=0.1):
    return FedSERClassifier(dropout=dropout)


def l2_penalty(model, coefficient=0.0001):
    """Keras L2 on the 12 block kernels; attention, biases and dense excluded.

    Add this scalar to the training loss. Optimizer weight decay also affects
    other parameters and does not reproduce the recovered source objective.
    """
    if coefficient < 0:
        raise ValueError("L2 coefficient must be nonnegative")
    return coefficient * sum(
        layer.weight.square().sum()
        for block in model.blocks
        for layer in block.modules()
        if isinstance(layer, nn.Conv2d)
    )
