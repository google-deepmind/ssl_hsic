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

"""Tests for kernels."""
from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from jax import random

from ssl_hsic import kernels


def inverse_multiquadratic(x, y, c):
  dist = kernels.pairwise_distance_square(x, y)
  return c / jnp.sqrt(dist + c ** 2)


def linear_kernel(x, y):
  dist = jnp.matmul(x, jnp.transpose(y))
  return dist


class KernelsTest(parameterized.TestCase):

  @parameterized.parameters((8,), (128,), (4096,))
  def test_get_label_weights(self, batch):
    x = jnp.eye(batch)
    k = linear_kernel(x, x)
    h = jnp.eye(batch) - 1/batch
    hkh = jnp.matmul(jnp.matmul(h, k), h)
    expected_maximum = hkh.max()
    expected_minimum = hkh.min()
    maximium, minimum = kernels.get_label_weights(batch)
    self.assertAlmostEqual(maximium, expected_maximum)
    self.assertAlmostEqual(minimum, expected_minimum)

  @parameterized.parameters((1, 1), (8, 1), (128, 1), (256, 1),
                            (1, 10), (8, 10), (128, 10), (256, 10),
                            (1, 0.1), (8, 0.1), (128, 0.1), (256, 0.1),)
  def test_imq_rff_features(self, dim, c):
    num_features = 512
    rng = random.PRNGKey(42)
    rng_x, rng_y, rng_rff = random.split(rng, 3)
    amp, amp_probs = kernels.imq_amplitude_frequency_and_probs(dim)
    x = random.uniform(rng_x, [1, dim])
    x_rff = kernels.imq_rff_features(num_features, rng_rff, x, c, amp,
                                     amp_probs)
    y = random.uniform(rng_y, [1, dim])
    y_rff = kernels.imq_rff_features(num_features, rng_rff, y, c, amp,
                                     amp_probs)

    expected = inverse_multiquadratic(x, y, c)
    rff_approx = jnp.matmul(x_rff, y_rff.T)
    self.assertAlmostEqual(rff_approx, expected, delta=0.1)

  @parameterized.parameters((1, 1), (8, 1), (128, 1), (256, 1),
                            (1, 10), (8, 10), (128, 10), (256, 10),
                            (1, 0.1), (8, 0.1), (128, 0.1), (256, 0.1),)
  def test_rff_approximate_hsic_xx(self, dim, c):
    num_features = 512
    rng = random.PRNGKey(42)
    rng1, rng2, rng_x = random.split(rng, 3)
    amp, amp_probs = kernels.imq_amplitude_frequency_and_probs(dim)
    rff_kwargs = {'amp': amp, 'amp_probs': amp_probs}
    x = random.uniform(rng_x, [1, dim])
    k = inverse_multiquadratic(x, x, c)
    k = k - jnp.mean(k, axis=1, keepdims=True)
    hsic_xx = (k * k).mean()
    hsic_xx_approx = kernels.rff_approximate_hsic_xx([x], num_features,
                                                     rng1, rng2, c, rff_kwargs)
    self.assertEqual(hsic_xx_approx, hsic_xx)

if __name__ == '__main__':
  absltest.main()
