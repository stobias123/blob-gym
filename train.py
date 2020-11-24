from __future__ import absolute_import, division, print_function

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image


from absl import app
from absl import flags
from absl import logging

import time
import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from blob_env.envs import *

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from gym.envs.registration import register
from gym.envs.registration import register

register(
    id='Blob2d-v1',
    entry_point='blob_env.envs.blob_env:BlobEnv')

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS

@gin.configurable
def train_eval(
    root_dir,
    env_name='Blob2d-v1',
    num_iterations=100000,
    train_sequence_length=1,
    collect_steps_per_iteration=1,
    initial_collect_steps =1500,
    replay_buffer_max_length = 10000,
    batch_size = 64,
    learning_rate=1e-3,
    num_eval_episodes=10,
    eval_interval=1000,
    # Params for QNetwork
    fc_layer_params = (100,),
    use_tf_functions = False,
    ## train params
    train_steps_per_iteration=1,
    train_checkpoint_interval=1000,
    policy_checkpoint_interval=1000,
    rb_checkpoint_interval=1000,
    n_step_update=1,
    ## Params for Summaries and logging
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()

  with tf.compat.v2.summary.record_if(
          lambda: tf.math.equal(global_step % summary_interval, 0)):
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
    eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

    if train_sequence_length != 1 and n_step_update != 1:
      raise NotImplementedError(
          'train_eval does not currently support n-step updates with stateful '
          'networks (i.e., RNNs)')
          
  env = suite_gym.load('Blob2d-v1')
  
  tf_env = tf_py_environment.TFPyEnvironment(env)

  action_spec = tf_env.action_spec()

  fc_layer_params = (100,)

  q_net = q_network.QNetwork(
      tf_env.observation_spec(),
      tf_env.action_spec(),
      fc_layer_params=fc_layer_params)

  agent = dqn_agent.DqnAgent(
      tf_env.time_step_spec(),
      tf_env.action_spec(),
      q_network=q_net,
      optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
      td_errors_loss_fn=common.element_wise_squared_loss,
      train_step_counter=global_step)
  agent.initialize()

  train_metrics = [
      tf_metrics.NumberOfEpisodes(),
      tf_metrics.EnvironmentSteps(),
      tf_metrics.AverageReturnMetric(),
      tf_metrics.AverageEpisodeLengthMetric(),
  ]

  eval_policy = agent.policy
  collect_policy = agent.collect_policy


  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=replay_buffer_max_length)

  collect_driver = dynamic_step_driver.DynamicStepDriver(
      tf_env,
      collect_policy,
      observers=[replay_buffer.add_batch] + train_metrics,
      num_steps=collect_steps_per_iteration)

  train_checkpointer = common.Checkpointer(
      ckpt_dir=train_dir,
      agent=agent,
      global_step=global_step,
      metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
  policy_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(train_dir, 'policy'),
      policy=eval_policy,
      global_step=global_step)
  rb_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
      max_to_keep=1,
      replay_buffer=replay_buffer)

  train_checkpointer.initialize_or_restore()
  rb_checkpointer.initialize_or_restore()

  initial_collect_policy = random_tf_policy.RandomTFPolicy(
      tf_env.time_step_spec(), tf_env.action_spec())

  logging.info(
      'Initializing replay buffer by collecting experience for %d steps with '
      'a random policy.', initial_collect_steps)
  dynamic_step_driver.DynamicStepDriver(
      tf_env,
      initial_collect_policy,
      observers=[replay_buffer.add_batch] + train_metrics,
      num_steps=initial_collect_steps).run()

  results = metric_utils.eager_compute(
      eval_metrics,
      eval_tf_env,
      eval_policy,
      num_episodes=num_eval_episodes,
      train_step=global_step,
      summary_writer=eval_summary_writer,
      summary_prefix='Metrics',
  )

  if eval_metrics_callback is not None:
      eval_metrics_callback(results, global_step.numpy())
  metric_utils.log_metrics(eval_metrics)

  time_step = None
  policy_state = collect_policy.get_initial_state(tf_env.batch_size)

  timed_at_step = global_step.numpy()
  time_acc = 0

  # Dataset generates trajectories with shape [Bx2x...]
  dataset = replay_buffer.as_dataset(
      num_parallel_calls=3,
      sample_batch_size=batch_size,
      num_steps=train_sequence_length + 1).prefetch(3)
  iterator = iter(dataset)

  def train_step():
    experience, _ = next(iterator)
    return agent.train(experience)

  if use_tf_functions:
    train_step = common.function(train_step)

  # Main Training loop.
  for _ in range(num_iterations):
    start_time = time.time()
    time_step, policy_state = collect_driver.run(
        time_step=time_step,
        policy_state=policy_state,
    )
    for _ in range(train_steps_per_iteration):
      train_loss = train_step()
    time_acc += time.time() - start_time

    if global_step.numpy() % log_interval == 0:
      logging.info('step = %d, loss = %f', global_step.numpy(),
                  train_loss.loss)
      steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
      logging.info('%.3f steps/sec', steps_per_sec)
      tf.compat.v2.summary.scalar(
          name='global_steps_per_sec', data=steps_per_sec, step=global_step)
      timed_at_step = global_step.numpy()
      time_acc = 0

    for train_metric in train_metrics:
      train_metric.tf_summaries(
          train_step=global_step, step_metrics=train_metrics[:2])

    if global_step.numpy() % train_checkpoint_interval == 0:
      train_checkpointer.save(global_step=global_step.numpy())

    if global_step.numpy() % policy_checkpoint_interval == 0:
      policy_checkpointer.save(global_step=global_step.numpy())

    if global_step.numpy() % rb_checkpoint_interval == 0:
      rb_checkpointer.save(global_step=global_step.numpy())

    if global_step.numpy() % eval_interval == 0:
      results = metric_utils.eager_compute(
          eval_metrics,
          eval_tf_env,
          eval_policy,
          num_episodes=num_eval_episodes,
          train_step=global_step,
          summary_writer=eval_summary_writer,
          summary_prefix='Metrics',
      )
      if eval_metrics_callback is not None:
        eval_metrics_callback(results, global_step.numpy())
      metric_utils.log_metrics(eval_metrics)
  return train_loss


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  train_eval(FLAGS.root_dir, num_iterations=FLAGS.num_iterations)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
