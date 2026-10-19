"""Pérdidas VAE: suma por píxel/latente y promedio por lote."""

import tensorflow as tf


def loss_terms(images, reconstruction, mean, logvar):
    r = tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(images, reconstruction), axis=(1, 2)
    )
    kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
    return r, kl


def sample_latent(mean, logvar, seed):
    noise = tf.random.stateless_normal(tf.shape(mean), seed=seed)
    return mean + tf.exp(0.5 * logvar) * noise
