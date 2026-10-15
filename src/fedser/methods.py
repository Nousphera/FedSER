"""Client-specific self-training and sample-weighted aggregation."""

import math
import numpy as np
import torch
from torch.nn import functional as F


def confidence_threshold(completed, client_completed, rounds, start=.5, end=.9, delta=.5):
    """Increasing cosine schedule, adjusted for each client's completed rounds.

    Follows the paper's increasing-threshold protocol; printed Eq. 3 omits
    the minimum offset and uses the opposite cosine sign.
    """
    if (rounds < 1 or not 0 <= client_completed <= completed <= rounds
            or not 0 <= start <= end <= 1 or not 0 <= delta <= 1):
        raise ValueError("Invalid confidence schedule")
    progress = completed - delta * (completed - client_completed)
    return start + (end - start) * (1 - math.cos(math.pi * progress / rounds)) / 2


def self_training_loss(logits, labels, threshold=.5, temperature=2., beta=.5):
    """Algorithm 1; temperature selects targets but never rescales training logits."""
    labeled = labels >= 0
    supervised = F.cross_entropy(logits[labeled], labels[labeled]) if labeled.any() else logits.sum() * 0
    unlabeled = logits[~labeled]
    if len(unlabeled):
        probabilities = (unlabeled.detach() / temperature).softmax(-1)
        confidence, targets = probabilities.max(-1)
        selected = confidence >= threshold
        unsupervised = (F.cross_entropy(unlabeled, targets, reduction="none") * selected).mean()
    else:
        selected = torch.empty(0, device=logits.device, dtype=torch.bool)
        unsupervised = logits.sum() * 0
    return supervised + beta * unsupervised, {
        "labeled-loss": float(supervised.detach()), "unlabeled-loss": float(unsupervised.detach()),
        "accepted": int(selected.sum()), "unlabeled": len(unlabeled),
    }


def aggregate(states, weights, reference):
    if not states or len(states) != len(weights):
        raise ValueError("Missing client updates")
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Positive finite client sample counts are required")
    for state in states:
        if set(state) != set(reference):
            raise ValueError("Client model keys differ")
        for key, value in state.items():
            if value.shape != reference[key].shape or value.dtype != reference[key].dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid client tensor: {key}")
    weights /= weights.sum()
    return {key: sum(state[key].double() * float(w) for state, w in zip(states, weights)).to(value.dtype)
            for key, value in reference.items()}
