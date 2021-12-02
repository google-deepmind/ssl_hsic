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

"""Helper functions for the experiments."""

import functools

import jax
from jax import numpy as jnp


def all_gather(value: jnp.ndarray) -> jnp.ndarray:
  """All-gather implementation with gradient.

  Args:
    value: the jnp.array to gather from all devices.

  Returns:
    [replica, *shape] jnp.array
  """
  num_devices = jax.device_count()
  bcast_shape = (num_devices,) + value.shape

  # Broadcast the value to the shape.
  value_bcasted = jnp.broadcast_to(value, bcast_shape)

  def all_to_all(x, concat_axis, split_axis):
    """Wrap the inner custom_gradient because it doesn't like non-array."""

    @jax.custom_gradient
    def _all_to_all(x):
      """Inner member that returns all-to-all and grad op."""
      # convenience shorten
      a2a = functools.partial(jax.lax.all_to_all, axis_name="i")

      def grad_fn(g):
        """Derivative of all_to_all is just concat and split axis swap."""
        return (a2a(g, split_axis=concat_axis,
                    concat_axis=split_axis),)

      # All-to-all is the closest op to all-gather within XLA.
      # returns a tuple of forward outputs and a backward fn.
      return a2a(x, split_axis=split_axis,
                 concat_axis=concat_axis), grad_fn

    return _all_to_all(x)

  # return the inner gradient function
  return all_to_all(value_bcasted, concat_axis=0, split_axis=0)
