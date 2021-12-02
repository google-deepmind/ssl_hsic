#!/usr/bin/python
#
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for computing SSL-HSIC losses."""

import functools
from typing import Any, Dict, List, Optional, Text, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import mpmath
import numpy as np


def pairwise_distance_square(x: jnp.ndarray,
                             y: jnp.ndarray, maximum=1e10) -> jnp.ndarray:
  """Computes the square of pairwise distances.

  dist_ij = (x[i] - y[j])'(x[i] - y[j])
          = x[i]'x[i] - 2x[i]'y[j] + y[j]'y[j]

  dist = x.x_{ij,ik->i} - 2 x.y_{ik,jk->ij} + y.y_{ij,ik->i}
  Args:
    x: tf.Tensor [B1, d].
    y: tf.Tensor [B2, d]. If y is None, then y=x.
    maximum: the maximum value to avoid overflow.

  Returns:
    Pairwise distance matrix [B1, B2].
  """
  x_sq = jnp.einsum('ij,ij->i', x, x)[:, jnp.newaxis]
  y_sq = jnp.einsum('ij,ij->i', y, y)[jnp.newaxis, :]
  x_y = jnp.einsum('ik,jk->ij', x, y)
  dist = x_sq + y_sq - 2 * x_y
  # Safe in case dist becomes negative.
  return jnp.minimum(jnp.maximum(dist, 0.0), maximum)


def get_label_weights(batch: int) -> Tuple[float, float]:
  """Returns the positive and negative weights of the label kernel matrix."""
  w_pos_base = jnp.atleast_2d(1.0)
  w_neg_base = jnp.atleast_2d(0.0)
  w_mean = (w_pos_base + w_neg_base * (batch - 1)) / batch
  w_pos_base -= w_mean
  w_neg_base -= w_mean
  w_mean = (w_pos_base + w_neg_base * (batch - 1)) / batch
  w_pos = w_pos_base - w_mean
  w_neg = w_neg_base - w_mean
  return w_pos[0, 0], w_neg[0, 0]


def compute_prob(n: int, x_range: np.ndarray) -> np.ndarray:
  """Compute the probablity to sample the random fourier features."""
  probs = [mpmath.besselk((n - 1) / 2, x) * mpmath.power(x, (n - 1) / 2)
           for x in x_range]
  normalized_probs = [float(p / sum(probs)) for p in probs]
  return np.array(normalized_probs)


def imq_amplitude_frequency_and_probs(n: int) -> Tuple[np.ndarray, np.ndarray]:
  """Returns the range and probablity for sampling RFF."""
  x = np.linspace(1e-12, 100, 10000)  # int(n * 10 / c)
  p = compute_prob(n, x)
  return x, p


def imq_rff_features(num_features: int, rng: jnp.DeviceArray, x: jnp.ndarray,
                     c: float, amp: jnp.ndarray,
                     amp_probs: jnp.ndarray) -> jnp.ndarray:
  """Returns the RFF feature for IMQ kernel with pre-computed amplitude prob."""
  d = x.shape[-1]
  rng1, rng2 = jax.random.split(rng)
  amp = jax.random.choice(rng1, amp, shape=[num_features, 1], p=amp_probs)
  directions = jax.random.normal(rng2, shape=(num_features, d))
  b = jax.random.uniform(rng2, shape=(1, num_features)) * 2 * jnp.pi
  w = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True) * amp
  z_x = jnp.sqrt(2 / num_features) * jnp.cos(jnp.matmul(x / c, w.T) + b)
  return z_x


def rff_approximate_hsic_xy(list_hiddens: List[jnp.ndarray], w: float,
                            num_features: int, rng: jnp.DeviceArray, c: float,
                            rff_kwargs: Dict[Text, jnp.ndarray]) -> jnp.ndarray:
  """RFF approximation of Unbiased HSIC(X, Y).

  Args:
    list_hiddens: a list of features.
    w: difference between max and min of the label Y's gram matrix.
    num_features: number of RFF features used for the approximation.
    rng: random seed used for sampling RFF features of the hiddens.
    c: parameter of the inverse multiquadric kernel.
    rff_kwargs: keyword arguments used for sampling frequencies.

  Returns:
    Approximation of HSIC(X, Y) where the kernel is inverse multiquadric kernel.
  """
  b, _ = list_hiddens[0].shape
  k = len(list_hiddens)
  rff_hiddens = jnp.zeros((b, num_features))
  mean = jnp.zeros((1, num_features))
  n_square = (b * k) ** 2
  for hidden in list_hiddens:
    rff_features = imq_rff_features(num_features, rng, hidden, c, **rff_kwargs)
    rff_hiddens += rff_features
    mean += rff_features.sum(0, keepdims=True)
  return w * ((rff_hiddens**2).sum() / (b * k * (k - 1)) -
              (mean**2).sum() / n_square)


def rff_approximate_hsic_xx(
    list_hiddens: List[jnp.ndarray], num_features: int, rng: jnp.DeviceArray,
    rng_used: jnp.DeviceArray, c: float, rff_kwargs: Dict[Text, jnp.ndarray]
    ) -> jnp.ndarray:
  """RFF approximation of HSIC(X, X) where inverse multiquadric kernel is used.

  Args:
    list_hiddens: a list of features.
    num_features: number of RFF features used for the approximation.
    rng: random seed used for sampling the first RFF features.
    rng_used: random seed used for sampling the second RFF features.
    c: parameter of the inverse multiquadric kernel.
    rff_kwargs: keyword arguments used for sampling frequencies.

  Returns:
    Approximation of HSIC(X, X) where the kernel is inverse multiquadric kernel.
  """
  x1_rffs = []
  x2_rffs = []

  for xs in list_hiddens:
    x1_rff = imq_rff_features(num_features, rng_used, xs, c, **rff_kwargs)
    x1_rffs.append(x1_rff)
    x2_rff = imq_rff_features(num_features, rng, xs, c, **rff_kwargs)
    x2_rffs.append(x2_rff)

  mean_x1 = (functools.reduce(jax.lax.add, x1_rffs) / len(x1_rffs)).mean(
      0, keepdims=True)
  mean_x2 = (functools.reduce(jax.lax.add, x2_rffs) / len(x2_rffs)).mean(
      0, keepdims=True)
  z = jnp.zeros(shape=(num_features, num_features), dtype=jnp.float32)
  for x1_rff, x2_rff in zip(x1_rffs, x2_rffs):
    z += jnp.einsum('ni,nj->ij', x1_rff - mean_x1, x2_rff - mean_x2)
  return (z ** 2).sum() / ((x1_rff.shape[0] * len(list_hiddens)) ** 2)


class HSICLoss(hk.Module):
  """SSL-HSIC loss."""

  def __init__(self,
               num_rff_features: int,
               regul_weight: float,
               name: Optional[Text] = 'hsic_loss'):
    """Initialize HSICLoss.

    Args:
      num_rff_features: number of RFF features used for the approximation.
      regul_weight: regularization weight applied for HSIC(X, X).
      name: name of the module, optional.
    """
    super().__init__(name=name)
    self._num_rff_features = num_rff_features
    self._regul_weight = regul_weight

  def __call__(
      self, list_hiddens: List[jnp.ndarray],
      rff_kwargs: Optional[Dict[Text, Any]]
      ) -> Tuple[jnp.ndarray, Dict[Text, jnp.ndarray]]:
    """Returns the HSIC loss and summaries.

    Args:
      list_hiddens: list of hiddens from different views.
      rff_kwargs: keyword args for sampling frequencies to compute RFF.

    Returns:
      total loss and a dictionary of summaries.
    """
    b = list_hiddens[0].shape[0]
    scale = hk.get_parameter('scale', shape=[], dtype=jnp.float32,
                             init=hk.initializers.Constant(1.))
    c = jax.lax.stop_gradient(scale)
    rff_kwargs = rff_kwargs or {}
    w_pos, w_neg = get_label_weights(b)
    rng1, rng2 = jax.random.split(hk.next_rng_key())
    hsic_xy = rff_approximate_hsic_xy(list_hiddens, w_pos - w_neg,
                                      self._num_rff_features, rng1, c,
                                      rff_kwargs=rff_kwargs)
    hsic_xx = rff_approximate_hsic_xx(list_hiddens, self._num_rff_features,
                                      rng1, rng2, c, rff_kwargs)
    total_loss = self._regul_weight * jnp.sqrt(hsic_xx) - hsic_xy

    # Compute gradient norm.
    n_samples = int(1024 / len(list_hiddens))  # 1024 samples in total.
    sampled_hiddens_1 = jnp.concatenate([
        x[jax.random.choice(hk.next_rng_key(), jnp.arange(b), (n_samples,)), :]
        for x in list_hiddens
    ])
    sampled_hiddens_2 = jnp.concatenate([
        x[jax.random.choice(hk.next_rng_key(), jnp.arange(b), (n_samples,)), :]
        for x in list_hiddens
    ])
    dist_sq = jax.lax.stop_gradient(
        pairwise_distance_square(sampled_hiddens_1, sampled_hiddens_2))
    grad = jax.grad(lambda x, y: (y / jnp.sqrt(x + y**2)).sum())(dist_sq, scale)
    grad_norm = 0.5 * jnp.log(jnp.maximum(1e-14, grad ** 2)).mean()
    summaries = {'kernel_loss/hsic_xy': hsic_xy,
                 'kernel_loss/hsic_xx': hsic_xx,
                 'kernel_loss/total_loss': total_loss,
                 'kernel_loss/kernel_param': scale,
                 'kernel_loss/grad_norm': grad_norm}
    return total_loss, summaries
