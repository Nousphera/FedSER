import numpy as np
import pytest
import tensorflow as tf

from fedser.method import confidence_threshold, self_training_loss, aggregate, unweighted_accuracy
from fedser.model import STCAttention, build_model


def test_participation_aware_cosine_endpoints_and_delayed_client():
    assert confidence_threshold(0, 0, 100) == pytest.approx(.5)
    assert confidence_threshold(100, 100, 100) == pytest.approx(.9)
    assert confidence_threshold(50, 10, 100) < confidence_threshold(50, 50, 100)
    assert confidence_threshold(90, 0, 100, mode="source") == pytest.approx(.5)
    with pytest.raises(ValueError):
        confidence_threshold(5, 6, 100)


def test_masked_loss_uses_all_unlabeled_denominator_and_detached_targets():
    logits = tf.Variable([[2., 0., 0., 0.], [8., 0., 0., 0.], [0., 0., 0., 0.]])
    with tf.GradientTape() as tape:
        loss, info = self_training_loss(logits, tf.constant([0, -1, -1]), .8, 2., .5)
    sup = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[0], logits=logits[:1])[0]
    unsup = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[0], logits=logits[1:2])[0] / 2
    assert float(loss) == pytest.approx(float(sup + .5 * unsup))
    assert info["accepted"] == 1
    np.testing.assert_array_equal(tape.gradient(loss, logits)[2], np.zeros(4))


def test_supervised_only_and_empty_pseudolabel_selection_are_finite():
    logits = tf.zeros((3, 4))
    loss, _ = self_training_loss(logits, [0, 1, 2], .9, 2., .5)
    assert float(loss) == pytest.approx(np.log(4))
    source, _ = self_training_loss(logits, [0, -1, -1], .9, 1., .5, mode="source")
    assert float(source) == pytest.approx(.5 * np.log(4))


def test_fedavg_weights_by_exact_utterance_counts_and_rejects_mismatch():
    result = aggregate([([np.array([1., 3.], np.float32)], 1), ([np.array([5., 7.], np.float32)], 3)])
    np.testing.assert_allclose(result[0], [4., 6.])
    with pytest.raises(ValueError):
        aggregate([([np.zeros(2)], 1), ([np.zeros(3)], 1)])


def test_ua_is_mean_recall_not_overall_accuracy():
    y = [0] * 8 + [1, 2, 3]
    assert unweighted_accuracy(y, [0] * 11) == .25


def test_attention_paper_equation_and_time_axis_normalization():
    x = tf.reshape(tf.range(1., 97.), (1, 4, 3, 8)) / 100.
    layer = STCAttention(mode="paper")
    layer(x)
    for variable in layer.trainable_variables:
        variable.assign(tf.ones_like(variable) * .1)
    pooled = tf.concat([tf.reduce_max(x, -1, keepdims=True), tf.reduce_mean(x, -1, keepdims=True)], -1)
    spectral = tf.reduce_sum(pooled, axis=(1, 3), keepdims=True) * .1 + .1
    temporal = tf.reduce_sum(pooled, axis=(2, 3), keepdims=True) * .1 + .1
    first = tf.reduce_sum(tf.reduce_mean(x, (1, 2)), -1) * .1 + .1
    channel = tf.tanh(first * 4 * .1 + .1)[:, None, None, None]
    expected = tf.nn.softmax(spectral * temporal * channel, axis=1)
    np.testing.assert_allclose(layer(x), tf.repeat(expected, 8, axis=-1), atol=1e-7)
    np.testing.assert_allclose(tf.reduce_sum(layer(x), axis=1), np.ones((1, 3, 8)), atol=1e-6)


@pytest.mark.parametrize("mode", ["paper", "source"])
def test_full_attention_network_updates_and_saves(tmp_path, mode):
    tf.keras.utils.set_random_seed(7)
    model = build_model((32, 32, 1), mode=mode)
    x = tf.random.normal((2, 32, 32, 1))
    before = [w.copy() for w in model.get_weights()]
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss, _ = self_training_loss(logits, [0, 1], .5, 2., .5)
        loss += tf.add_n(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None and np.all(np.isfinite(g)) for g in gradients)
    tf.keras.optimizers.Adam(.001).apply_gradients(zip(gradients, model.trainable_variables))
    assert any(not np.array_equal(a, b) for a, b in zip(before, model.get_weights()))
    assert logits.shape == (2, 4)
    path = tmp_path / "model.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(restored(x), model(x), atol=1e-6)
