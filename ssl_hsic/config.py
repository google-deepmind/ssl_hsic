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

"""Config file for SSL-HSIC experiment."""

from typing import Text

from absl import logging
from byol.utils import dataset
from jaxline import base_config
from ml_collections import config_dict

CONFIG = {
    'default': (1000, 4096, '/tmp/ssl_hsic_final'),
    'test': (0.00001, 3, '/tmp/ssl_hsic_final')
}


def get_config(config_key: Text = 'test') -> config_dict.ConfigDict:
  """Return config object, containing all hyperparameters for training."""
  num_epochs, batch_size, save_dir = CONFIG[config_key]
  train_images_per_epoch = dataset.Split.TRAIN_AND_VALID.num_examples
  config = base_config.get_base_config()
  config.experiment_kwargs = config_dict.ConfigDict(dict(
      num_classes=1000,
      batch_size=batch_size,
      max_steps=int(num_epochs * train_images_per_epoch // batch_size),
      enable_double_transpose=True,
      base_target_ema=0.99,
      ema_decay=0.999,
      network_config=config_dict.ConfigDict(dict(
          projector_hidden_size=4096,
          projector_output_size=256,
          predictor_hidden_size=4096,
          encoder_class='ResNet50',  # Should match a class in utils/networks.
          encoder_config=config_dict.ConfigDict(dict(
              resnet_v2=False,
              width_multiplier=1)),
          bn_config={
              'decay_rate': .9,
              'eps': 1e-5,
              # Accumulate batchnorm statistics across devices.
              # This should be equal to the `axis_name` argument passed
              # to jax.pmap.
              'cross_replica_axis': 'i',
              'create_scale': True,
              'create_offset': True,
          })),
      loss_config=config_dict.ConfigDict(dict(
          num_rff_features=512,
          regul_weight=3,
      )),
      optimizer_config=config_dict.ConfigDict(dict(
          weight_decay=1e-6,
          eta=1e-3,
          momentum=.9,
      )),
      lr_schedule_config=dict(
          base_learning_rate=0.4,
          warmup_steps=10 * train_images_per_epoch // batch_size,
      ),
      evaluation_config=config_dict.ConfigDict(dict(
          subset='test',
          batch_size=100,
      )),
      save_dir=save_dir,  # Save the last checkpoint for evaluation.
  ))
  config.checkpoint_dir = '/tmp/ssl_hsic'
  config.train_checkpoint_all_hosts = False
  config.training_steps = config.experiment_kwargs.max_steps
  logging.info(config)
  return config
