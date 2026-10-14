"""Self-training and aggregation; see docs/method.md for source deviations."""
import math
import numpy as np
import tensorflow as tf


def confidence_threshold(completed, client_completed, total, minimum=.5, maximum=.9,
                         delta=.5, mode="paper"):
    if not (total > 0 and 0 <= client_completed <= completed <= total
            and 0 <= minimum <= maximum <= 1 and 0 <= delta <= 1):
        raise ValueError("Invalid round counters or confidence limits")
    if mode not in {"paper", "source"}:
        raise ValueError("mode must be paper or source")
    progress = client_completed if mode == "source" else completed - delta * (completed - client_completed)
    return minimum + (maximum - minimum) * .5 * (1 - math.cos(math.pi * progress / total))


def self_training_loss(logits, labels, threshold, temperature, beta, mode="paper"):
    """Mean labeled CE plus masked unlabeled CE / total unlabeled count."""
    if not (temperature > 0 and 0 <= threshold <= 1 and 0 <= beta <= 1):
        raise ValueError("Invalid self-training hyperparameters")
    if mode not in {"paper", "source"}:
        raise ValueError("mode must be paper or source")
    labels = tf.convert_to_tensor(labels, tf.int32)
    tf.debugging.assert_greater_equal(labels, -1)
    tf.debugging.assert_less(labels, tf.shape(logits)[1])
    labeled = labels >= 0
    probabilities = tf.stop_gradient(tf.nn.softmax(logits / temperature))
    targets = tf.argmax(probabilities, axis=1, output_type=tf.int32)
    accepted = tf.logical_and(~labeled, tf.reduce_max(probabilities, axis=1) >= threshold)
    sup_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.maximum(labels, 0), logits=logits)
    unsup_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
    sup = tf.math.divide_no_nan(tf.reduce_sum(tf.where(labeled, sup_ce, 0.)),
                               tf.reduce_sum(tf.cast(labeled, logits.dtype)))
    unsup = tf.math.divide_no_nan(tf.reduce_sum(tf.where(accepted, unsup_ce, 0.)),
                                 tf.reduce_sum(tf.cast(~labeled, logits.dtype)))
    coefficient = 1 - beta if mode == "source" else 1.
    return coefficient * sup + beta * unsup, {
        "supervised_loss": sup, "unsupervised_loss": unsup,
        "accepted": tf.reduce_sum(tf.cast(accepted, tf.int32)),
        "unlabeled": tf.reduce_sum(tf.cast(~labeled, tf.int32)),
    }


def aggregate(updates):
    """FedAvg over exact utterance counts, with structural validation."""
    if not updates or any(n <= 0 for _, n in updates):
        raise ValueError("FedAvg needs nonempty positive-weight updates")
    reference = updates[0][0]
    for weights, _ in updates:
        if len(weights) != len(reference) or any(a.shape != b.shape for a, b in zip(weights, reference)):
            raise ValueError("Model parameter structures differ")
        if any(not np.isfinite(a).all() for a in weights):
            raise ValueError("Nonfinite client weights")
    total = sum(n for _, n in updates)
    return [sum(w[i].astype(np.float64) * (n / total) for w, n in updates).astype(reference[i].dtype)
            for i in range(len(reference))]


def unweighted_accuracy(labels, predictions, classes=4):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.size == 0 or labels.shape != predictions.shape:
        raise ValueError("Evaluation labels/predictions must be nonempty and aligned")
    if any(not np.any(labels == c) for c in range(classes)):
        raise ValueError("UA requires all four reference classes in the evaluation set")
    return float(np.mean([np.mean(predictions[labels == c] == c) for c in range(classes)]))
