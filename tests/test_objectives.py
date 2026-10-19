import numpy as np
import tensorflow as tf
from inf8239_u03_gen.objectives import loss_terms, sample_latent


def test_reconstruction_sum_and_kl_reductions():
    x = tf.zeros((2, 28, 28, 1))
    y = tf.fill((2, 28, 28, 1), 0.5)
    mu = tf.zeros((2, 2))
    lv = tf.zeros((2, 2))
    r, k = loss_terms(x, y, mu, lv)
    assert np.allclose(r.numpy(), 784 * np.log(2), rtol=1e-5)
    assert np.allclose(k.numpy(), 0)


def test_sampling_reproducible_and_gradient_flows():
    m = tf.Variable([[0.2, 0.3]])
    lv = tf.Variable([[0.0, 0.0]])
    with tf.GradientTape() as tape:
        z = sample_latent(m, lv, [42, 1])
        loss = tf.reduce_sum(z)
    grads = tape.gradient(loss, [m, lv])
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in grads)
    assert np.array_equal(z.numpy(), sample_latent(m, lv, [42, 1]).numpy())
