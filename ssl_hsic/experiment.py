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

"""SSL-HSIC pre-training."""

import functools
import os
from typing import Any, Generator, Mapping, NamedTuple, Optional, Text, Tuple

from absl import app
from absl import flags
from absl import logging
from acme.jax import utils as acme_utils
from byol.utils import augmentations
from byol.utils import dataset
from byol.utils import helpers as byol_helpers
from byol.utils import networks
from byol.utils import optimizers
from byol.utils import schedules
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import experiment
from jaxline import platform
from jaxline import utils as pipeline_utils
import numpy as np
import optax
from ssl_hsic import helpers
from ssl_hsic import kernels
import tensorflow as tf


# Type declarations.
LogsDict = Mapping[Text, jnp.ndarray]


class _ExperimentState(NamedTuple):
  """model and optimization parameters and state."""
  online_params: hk.Params
  target_params: hk.Params
  online_state: hk.State
  target_state: hk.State
  kernel_params: hk.Params
  kernel_opt_state: optax.OptState
  opt_state: optimizers.LarsState


class _EMAState(NamedTuple):
  """EMA parameters and state."""
  online_params: hk.Params
  kernel_params: hk.Params
  ema_params: hk.Params
  ema_state: hk.State


class Experiment(experiment.AbstractExperiment):
  """Training and evaluation for SSL-HSIC experiment."""

  # Holds a map from object properties that will be checkpointed to their name
  # within a checkpoint. Currently it is assumed that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {
      '_experiment_state': 'experiment_state',
      '_ema_state': 'ema_state',
  }

  def __init__(
      self,
      mode: Text,
      init_rng: jnp.DeviceArray,
      num_classes: int,
      batch_size: int,
      max_steps: int,
      enable_double_transpose: bool,
      base_target_ema: float,
      ema_decay: float,
      network_config: Mapping[Text, Any],
      loss_config: Mapping[Text, Any],
      optimizer_config: Mapping[Text, Any],
      lr_schedule_config: Mapping[Text, Any],
      evaluation_config: Mapping[Text, Any],
      save_dir: Optional[Text]):
    """Constructs the experiment.

    Args:
      mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
      init_rng: A `PRNGKey` to use for experiment initialization.
      num_classes: the number of classes; used for the online evaluation.
      batch_size: the total batch size; should be a multiple of the number of
        available accelerators.
      max_steps: the number of training steps; used for the lr/target network
        ema schedules.
      enable_double_transpose: see dataset.py; only has effect on TPU.
      base_target_ema: the initial value for the ema decay rate of the target
        network.
      ema_decay: the ema decay rate for the model parameters.
      network_config: the configuration for the network.
      loss_config: the configuration for the SSL-HSIC loss.
      optimizer_config: the configuration for the optimizer.
      lr_schedule_config: the configuration for the learning rate schedule.
      evaluation_config: the evaluation configuration.
      save_dir: directory to save the last checkpoint if provided.
    """
    super().__init__(mode, init_rng)
    self._init_rng = init_rng
    self._num_classes = num_classes
    self._lr_schedule_config = lr_schedule_config
    self._batch_size = batch_size
    self._max_steps = max_steps
    self._base_target_ema = base_target_ema
    self._optimizer_config = optimizer_config
    self._evaluation_config = evaluation_config
    self._ema_decay = ema_decay
    self._save_dir = save_dir
    # Checkpointed experiment state.
    self._experiment_state = None
    self._ema_state = None

    # Input pipelines.
    self._train_input = None
    self._eval_input = None
    self._should_transpose_images = (
        enable_double_transpose
        and jax.local_devices()[0].platform == 'tpu')
    # build the transformed ops
    forward_fn = functools.partial(self._forward, **network_config)
    self.forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    imq_amp, imq_amp_prob = kernels.imq_amplitude_frequency_and_probs(
        network_config['projector_output_size'])
    self._rff_kwargs = {'amp': imq_amp, 'amp_probs': imq_amp_prob}
    self.kernel_loss_fn = hk.transform(
        lambda *args: kernels.HSICLoss(**loss_config)(*args))  # pylint: disable=unnecessary-lambda

    # EMA of parameters.
    self.ema_update = hk.without_apply_rng(
        hk.transform_with_state(self._apply_ema_update))

    # training can handle multiple devices, thus the pmap
    self.update_pmap = jax.pmap(self._update_fn, axis_name='i')
    # evaluation can only handle single device
    self.eval_batch_jit = jax.jit(self._eval_batch)

    # Optimizer for kernel scale.
    self.kernel_optimizer = optax.adam(3e-4, b1=0.9, b2=0.999, eps=1e-8)

  def _apply_ema_update(self, params: hk.Params) -> hk.Params:
    ema = jax.tree_map(
        lambda x: hk.ExponentialMovingAverage(self._ema_decay),
        params)
    return jax.tree_multimap(lambda e, p: e(p), ema, params)

  def _forward(
      self,
      inputs: dataset.Batch,
      projector_hidden_size: int,
      projector_output_size: int,
      predictor_hidden_size: int,
      encoder_class: Text,
      encoder_config: Mapping[Text, Any],
      bn_config: Mapping[Text, Any],
      is_training: bool,
  ) -> Mapping[Text, jnp.ndarray]:
    """Forward application of architecture.

    Args:
      inputs: A batch of data, i.e. a dictionary, with either two keys,
        (`images` and `labels`) or three keys (`view1`, `view2`, `labels`).
      projector_hidden_size: hidden size of the projector MLP.
      projector_output_size: output size of the projector and predictor MLPs.
      predictor_hidden_size: hidden size of the predictor MLP.
      encoder_class: type of the encoder (should match a class in
        utils/networks).
      encoder_config: passed to the encoder constructor.
      bn_config: passed to the hk.BatchNorm constructors.
      is_training: Training or evaluating the model? When True, inputs must
        contain keys `view1` and `view2`. When False, inputs must contain key
        `images`.

    Returns:
      All outputs of the model, i.e. a dictionary with projection, prediction
      and logits keys, for either the two views, or the image.
    """
    encoder = getattr(networks, encoder_class)
    net = encoder(
        num_classes=None,  # Don't build the final linear layer
        bn_config=bn_config,
        **encoder_config)

    projector = networks.MLP(
        name='projector',
        hidden_size=projector_hidden_size,
        output_size=projector_output_size,
        bn_config=bn_config)
    predictor = networks.MLP(
        name='predictor',
        hidden_size=predictor_hidden_size,
        output_size=projector_output_size,
        bn_config=bn_config)
    classifier = hk.Linear(
        output_size=self._num_classes, name='classifier')

    def apply_once_fn(images: jnp.ndarray, suffix: Text = ''):
      images = dataset.normalize_images(images)

      embedding = net(images, is_training=is_training)
      proj_out = projector(embedding, is_training)
      pred_out = predictor(proj_out, is_training)

      # Note the stop_gradient: label information is not leaked into the
      # main network.
      classif_out = classifier(jax.lax.stop_gradient(embedding))
      outputs = {}
      outputs['projection' + suffix] = proj_out
      outputs['prediction' + suffix] = pred_out
      outputs['logits' + suffix] = classif_out
      return outputs

    if is_training:
      outputs_view1 = apply_once_fn(inputs['view1'], '_view1')
      outputs_view2 = apply_once_fn(inputs['view2'], '_view2')
      return {**outputs_view1, **outputs_view2}
    else:
      return apply_once_fn(inputs['images'], '')

  def _optimizer(self, learning_rate: float) -> optax.GradientTransformation:
    """Build optimizer from config."""
    return optimizers.lars(
        learning_rate,
        weight_decay_filter=optimizers.exclude_bias_and_norm,
        lars_adaptation_filter=optimizers.exclude_bias_and_norm,
        **self._optimizer_config)

  def loss_fn(
      self,
      online_params: hk.Params,
      target_params: hk.Params,
      kernel_params: hk.Params,
      online_state: hk.State,
      target_state: hk.Params,
      rng: jnp.ndarray,
      inputs: dataset.Batch,
  ) -> Tuple[jnp.ndarray, Tuple[Mapping[Text, hk.State], LogsDict]]:
    """Compute SSL-HSIC loss function.

    Args:
      online_params: parameters of the online network (the loss is later
        differentiated with respect to the online parameters).
      target_params: parameters of the target network.
      kernel_params: parameters of the kernel loss.
      online_state: internal state of online network.
      target_state: internal state of target network.
      rng: random number generator state.
      inputs: inputs, containing two batches of crops from the same images,
        view1 and view2 and labels

    Returns:
      SSL-HSIC loss, a mapping containing the online and target networks updated
      states after processing inputs, and various logs.
    """
    rng_aug, rng_hsic = jax.random.split(rng)
    rff_kwargs = inputs['rff_kwargs']
    if self._should_transpose_images:
      inputs = dataset.transpose_images(inputs)
    inputs = augmentations.postprocess(inputs, rng_aug)
    labels = inputs['labels']

    online_network_out, online_state = self.forward.apply(
        params=online_params,
        state=online_state,
        inputs=inputs,
        is_training=True)
    target_network_out, target_state = self.forward.apply(
        params=target_params,
        state=target_state,
        inputs=inputs,
        is_training=True)

    # Representation loss.
    hiddens = [online_network_out['prediction_view1'],
               online_network_out['prediction_view2'],
               jax.lax.stop_gradient(target_network_out['projection_view2']),
               jax.lax.stop_gradient(target_network_out['projection_view1'])]
    hiddens = [byol_helpers.l2_normalize(h, axis=-1) for h in hiddens]
    if jax.device_count() > 1:
      feature_dim = hiddens[0].shape[-1]
      hiddens = [
          helpers.all_gather(h).reshape(-1, feature_dim) for h in hiddens
      ]
    hsic_loss, summaries = self.kernel_loss_fn.apply(kernel_params, rng_hsic,
                                                     hiddens, rff_kwargs)
    grad_norm_loss = -summaries['kernel_loss/grad_norm']

    # Classification loss (with gradient flows stopped from flowing into the
    # ResNet). This is used to provide an evaluation of the representation
    # quality during training.
    classif_loss = byol_helpers.softmax_cross_entropy(
        logits=online_network_out['logits_view1'],
        labels=jax.nn.one_hot(labels, self._num_classes))

    top1_correct = byol_helpers.topk_accuracy(
        online_network_out['logits_view1'],
        inputs['labels'],
        topk=1,
    )

    top5_correct = byol_helpers.topk_accuracy(
        online_network_out['logits_view1'],
        inputs['labels'],
        topk=5,
    )

    top1_acc = jnp.mean(top1_correct)
    top5_acc = jnp.mean(top5_correct)

    classif_loss = jnp.mean(classif_loss)
    loss = hsic_loss + grad_norm_loss + classif_loss
    logs = dict(
        loss=loss,
        hsic_loss=hsic_loss,
        grad_norm_loss=grad_norm_loss,
        classif_loss=classif_loss,
        top1_accuracy=top1_acc,
        top5_accuracy=top5_acc,
        **summaries,
    )
    return loss, (dict(online_state=online_state,
                       target_state=target_state), logs)

  def _update_fn(
      self,
      experiment_state: _ExperimentState,
      ema_state: _EMAState,
      global_step: jnp.ndarray,
      rng: jnp.ndarray,
      inputs: dataset.Batch,
  ) -> Tuple[_ExperimentState, _EMAState, LogsDict]:
    """Update online and target parameters.

    Args:
      experiment_state: current network parameters and state.
      ema_state: ema parameters and state.
      global_step: current training step.
      rng: current random number generator
      inputs: inputs, containing two batches of crops from the same images,
        view1 and view2 and labels

    Returns:
      Tuple containing the updated experiment state after processing the inputs,
      EMA state after applying EMA update and various logs.
    """
    online_params = experiment_state.online_params
    target_params = experiment_state.target_params
    online_state = experiment_state.online_state
    target_state = experiment_state.target_state
    opt_state = experiment_state.opt_state
    kernel_params = experiment_state.kernel_params
    kernel_opt_state = experiment_state.kernel_opt_state

    # update online network
    grad_fn = jax.grad(self.loss_fn, argnums=[0, 2], has_aux=True)
    grads, (net_states, logs) = grad_fn(online_params, target_params,
                                        kernel_params,
                                        online_state, target_state,
                                        rng, inputs)

    # cross-device grad and logs reductions
    grads = jax.tree_map(lambda v: jax.lax.pmean(v, axis_name='i'), grads)
    logs = jax.tree_multimap(lambda x: jax.lax.pmean(x, axis_name='i'), logs)

    learning_rate = schedules.learning_schedule(
        global_step,
        batch_size=self._batch_size,
        total_steps=self._max_steps,
        **self._lr_schedule_config)
    grads, kernel_grads = grads
    updates, opt_state = self._optimizer(learning_rate).update(
        grads, opt_state, online_params)
    online_params = optax.apply_updates(online_params, updates)

    # update target network
    tau = schedules.target_ema(
        global_step,
        base_ema=self._base_target_ema,
        max_steps=self._max_steps)
    target_params = jax.tree_multimap(lambda x, y: x + (1 - tau) * (y - x),
                                      target_params, online_params)
    logs['tau'] = tau
    logs['learning_rate'] = learning_rate

    # Update kernel parameters.
    kernel_updates, kernel_opt_state = self.kernel_optimizer.update(
        kernel_grads, kernel_opt_state)
    kernel_params = optax.apply_updates(kernel_params, kernel_updates)

    # Update EMA parameters.
    (ema_online_params,
     ema_kernel_params), ema_module_state = self.ema_update.apply(
         ema_state.ema_params, ema_state.ema_state,
         [online_params, kernel_params])
    ema_state = _EMAState(
        online_params=ema_online_params,
        kernel_params=ema_kernel_params,
        ema_params=ema_state.ema_params,
        ema_state=ema_module_state)
    return _ExperimentState(
        online_params=online_params,
        target_params=target_params,
        online_state=net_states['online_state'],
        target_state=net_states['target_state'],
        opt_state=opt_state,
        kernel_params=kernel_params,
        kernel_opt_state=kernel_opt_state), ema_state, logs

  def _make_initial_state(
      self,
      rng: jnp.ndarray,
      dummy_input: dataset.Batch,
  ) -> Tuple[_ExperimentState, _EMAState]:
    """Initializate the experiment state and EMA state.

    Args:
      rng: random number generator used to initialize parameters. If working in
        a multi device setup, this need to be a ShardedArray.
      dummy_input: a dummy image, used to compute intermediate outputs shapes.

    Returns:
      Initial experiment state and EMA state.
    """
    rng_online, rng_target, rng_kernel, rng_ema = jax.random.split(rng, 4)

    if self._should_transpose_images:
      dummy_input = dataset.transpose_images(dummy_input)

    # Online and target parameters are initialized using different rngs,
    # in our experiments we did not notice a significant different with using
    # the same rng for both.
    online_params, online_state = self.forward.init(
        rng_online,
        dummy_input,
        is_training=True,
    )
    target_params, target_state = self.forward.init(
        rng_target,
        dummy_input,
        is_training=True,
    )
    opt_state = self._optimizer(0).init(online_params)

    out, _ = self.forward.apply(online_params, online_state, dummy_input,
                                is_training=True)
    # Init kernel loss. It doesn't matter which hiddens to take.
    hiddens = [
        out['prediction_view1'], out['prediction_view2']
    ]
    kernel_params = self.kernel_loss_fn.init(rng_kernel, hiddens,
                                             dummy_input['rff_kwargs'])
    kernel_opt_state = self.kernel_optimizer.init(kernel_params)
    ema_module_params, ema_module_state = self.ema_update.init(
        rng_ema, [online_params, kernel_params])
    (ema_online_params,
     ema_kernel_params), ema_module_state = self.ema_update.apply(
         ema_module_params, ema_module_state, [online_params, kernel_params])
    ema_state = _EMAState(
        online_params=ema_online_params,
        kernel_params=ema_kernel_params,
        ema_params=ema_module_params,
        ema_state=ema_module_state)
    return _ExperimentState(
        online_params=online_params,
        target_params=target_params,
        opt_state=opt_state,
        online_state=online_state,
        target_state=target_state,
        kernel_params=kernel_params,
        kernel_opt_state=kernel_opt_state,
    ), ema_state

  def step(self, *,
           global_step: jnp.ndarray,
           rng: jnp.ndarray, **unused_kwargs) -> Mapping[Text, np.ndarray]:
    """Performs a single training step."""
    if self._train_input is None:
      self._initialize_train()

    inputs = next(self._train_input)

    self._experiment_state, self._ema_state, scalars = self.update_pmap(
        self._experiment_state,
        self._ema_state,
        global_step=global_step,
        rng=rng,
        inputs=inputs,
    )
    if self._save_dir and (jax.host_id() == 0):
      global_step_value = pipeline_utils.get_first(global_step)
      if global_step_value == self._max_steps - 1:
        f_np = lambda x: np.array(jax.device_get(pipeline_utils.get_first(x)))
        np_experiment_state = jax.tree_map(f_np, self._experiment_state)
        np_ema_state = jax.tree_map(f_np, self._ema_state)
        path_npy = os.path.join(self._save_dir, 'checkpoint.npy')
        with tf.io.gfile.GFile(path_npy, 'wb') as fp:
          np.save(fp, {'experiment_state': np_experiment_state._asdict(),
                       'ema_state': np_ema_state._asdict()})
        logging.info('Saved final checkpoint at %s', path_npy)
    return pipeline_utils.get_first(scalars)

  def _initialize_train(self):
    """Initialize for training."""
    self._train_input = acme_utils.prefetch(self._build_train_input())

    # Check we haven't already restored params
    if self._experiment_state is None:
      logging.info(
          'Initializing parameters rather than restoring from checkpoint.')

      # initialize parameters, ema parameters and setup optimizer state
      inputs = next(self._train_input)
      init_fn = jax.pmap(self._make_initial_state, axis_name='i')

      # Init uses the same RNG key on all hosts+devices to ensure everyone
      # computes the same initial state and parameters.
      init_rng = byol_helpers.bcast_local_devices(self._init_rng)
      self._experiment_state, self._ema_state = init_fn(rng=init_rng,
                                                        dummy_input=inputs)

  def _build_train_input(self) -> Generator[dataset.Batch, None, None]:
    """Loads the (infinitely looping) dataset iterator."""
    num_devices = jax.device_count()
    global_batch_size = self._batch_size
    per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

    if ragged:
      raise ValueError(
          f'Global batch size {global_batch_size} must be divisible by '
          f'num devices {num_devices}')

    ds_numpy = dataset.load(
        dataset.Split.TRAIN_AND_VALID,
        preprocess_mode=dataset.PreprocessMode.PRETRAIN,
        transpose=self._should_transpose_images,
        batch_dims=[jax.local_device_count(), per_device_batch_size])

    # Add rff_kwargs to dataset, so that we don't build it as constant in the
    # graph.
    for inputs in ds_numpy:
      # pytype: disable=unsupported-operands
      inputs['rff_kwargs'] = jax.tree_map(
          lambda x: np.tile(x, (jax.local_device_count(), 1)), self._rff_kwargs)
      yield inputs

  def _eval_batch(
      self,
      params: hk.Params,
      state: hk.State,
      batch: dataset.Batch,
  ) -> Mapping[Text, jnp.ndarray]:
    """Evaluates a batch.

    Args:
      params: Parameters of the model to evaluate. Typically EMA parameters.
      state: State of the model to evaluate. Typically online state.
      batch: Batch of data to evaluate (must contain keys images and labels).

    Returns:
      Unreduced evaluation loss and top1 accuracy on the batch.
    """
    if self._should_transpose_images:
      batch = dataset.transpose_images(batch)

    outputs, _ = self.forward.apply(params, state, batch, is_training=False)
    logits = outputs['logits']
    labels = hk.one_hot(batch['labels'], self._num_classes)
    loss = byol_helpers.softmax_cross_entropy(logits, labels, reduction=None)
    top1_correct = byol_helpers.topk_accuracy(logits, batch['labels'], topk=1)
    top5_correct = byol_helpers.topk_accuracy(logits, batch['labels'], topk=5)
    # NOTE: Returned values will be summed and finally divided by num_samples.
    return {
        'eval_loss': loss,
        'top1_accuracy': top1_correct,
        'top5_accuracy': top5_correct,
    }

  def _eval_epoch(self, subset: Text, batch_size: int):
    """Evaluates an epoch."""
    num_samples = 0.
    summed_scalars = None
    params = pipeline_utils.get_first(self._ema_state.online_params)
    state = pipeline_utils.get_first(self._experiment_state.online_state)
    split = dataset.Split.from_string(subset)

    dataset_iterator = dataset.load(
        split,
        preprocess_mode=dataset.PreprocessMode.EVAL,
        transpose=self._should_transpose_images,
        batch_dims=[batch_size])

    for inputs in dataset_iterator:
      num_samples += inputs['labels'].shape[0]
      scalars = self.eval_batch_jit(params, state, inputs)

      # Accumulate the sum of scalars for each step.
      scalars = jax.tree_map(lambda x: jnp.sum(x, axis=0), scalars)
      if summed_scalars is None:
        summed_scalars = scalars
      else:
        summed_scalars = jax.tree_multimap(jnp.add, summed_scalars, scalars)

    mean_scalars = jax.tree_map(lambda x: x / num_samples, summed_scalars)
    return mean_scalars

  def evaluate(self, global_step: jnp.ndarray, **unused_args):
    """Thin wrapper around _eval_epoch."""

    global_step = np.array(pipeline_utils.get_first(global_step))
    scalars = jax.device_get(self._eval_epoch(**self._evaluation_config))

    logging.info('[Step %d] Eval scalars: %s', global_step, scalars)
    return scalars

if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, Experiment))
