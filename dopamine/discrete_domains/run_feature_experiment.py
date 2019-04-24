
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.agents.softmax_dqn import softmax_dqn_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger
from dopamine.discrete_domains.run_experiment import Runner

import numpy as np
import tensorflow as tf
import itertools
import gin.tf



def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 debug_mode=False):
  """Creates an agent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn':
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'softmax_dqn':
    return softmax_dqn_agent.SoftMaxDQNAgent(
      sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
def create_runner(base_dir, schedule='feature_train_and_eval'):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return Runner(base_dir, create_agent)
  elif schedule == 'feature_train_and_eval':
    return FeatureRunner(base_dir, create_agent)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return TrainRunner(base_dir, create_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))
@gin.configurable
class FeatureRunner(Runner):
    def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000,
               grid=False):
        Runner.__init__(self,
            base_dir=base_dir,
            create_agent_fn=create_agent_fn,
            create_environment_fn=create_environment_fn,
               checkpoint_file_prefix=checkpoint_file_prefix,
               logging_file_prefix=logging_file_prefix,
               log_every_n=log_every_n,
               num_iterations=num_iterations,
               training_steps=training_steps,
               evaluation_steps=evaluation_steps,
               max_steps_per_episode=max_steps_per_episode)

        if grid:
            height = self._environment.height
            width = self._environment.width
            num_actions = self._agent.num_actions
            states = list(itertools.product(range(height), range(width)))
            features = []
            states=np.array(states)
            net_outputs_all_states= self._agent.online_convnet(np.reshape(states, [height*width,2,1,1]))
            feature_tensor = net_outputs_all_states.features
            normalizer = height*width
        else:
            net_outputs_all_states = self._agent._replay_net_outputs
            batch_size=128
            feature_tensor = net_outputs_all_states.features
            normalizer = batch_size

        
        
        q_tensor = net_outputs_all_states.q_values
        s = tf.linalg.svd(feature_tensor, compute_uv=False)
        self.feature_histogram = tf.summary.histogram('singular_values', s)
        self.q_histogram = tf.summary.histogram('q_values', q_tensor)
        nonzeros = tf.count_nonzero(feature_tensor, axis=1)
        self.av_nonzero_entries= tf.summary.scalar('av_nonzeros', tf.reduce_mean(nonzeros))


    
    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction.

        An iteration involves running several episodes until a certain number of
        steps are obtained. The interleaving of train/eval phases implemented here
        are to match the implementation of (Mnih et al., 2015).

        Args:
          iteration: int, current iteration number, used as a global_step for saving
            Tensorboard summaries.

        Returns:
          A dict containing summary statistics for this iteration.
        """
        statistics = iteration_statistics.IterationStatistics()
        tf.logging.info('Starting iteration %d', iteration)
        num_episodes_train, average_reward_train = self._run_train_phase(
            statistics)
        num_episodes_eval, average_reward_eval = self._run_eval_phase(
            statistics)
        
        # print(self._sess.run(self._agent._replay_net_features, {self._agent.state_ph: self._agent.state}))

        self._save_tensorboard_summaries(iteration, num_episodes_train,
                                         average_reward_train, num_episodes_eval,
                                         average_reward_eval, sv_summary=self.feature_histogram)
        return statistics.data_lists
    def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  num_episodes_eval,
                                  average_reward_eval,
                                  sv_summary=None):
        """Save statistics as tensorboard summaries.

        Args:
          iteration: int, The current iteration number.
          num_episodes_train: int, number of training episodes run.
          average_reward_train: float, The average training reward.
          num_episodes_eval: int, number of evaluation episodes run.
          average_reward_eval: float, The average evaluation reward.
        """

        # include the histogram summaries
        h_summaries=tf.summary.merge_all()
        summ = self._sess.run(h_summaries)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Train/NumEpisodes',
                             simple_value=num_episodes_train),
            tf.Summary.Value(tag='Train/AverageReturns',
                             simple_value=average_reward_train),
            tf.Summary.Value(tag='Eval/NumEpisodes',
                             simple_value=num_episodes_eval),
            tf.Summary.Value(tag='Eval/AverageReturns',
                             simple_value=average_reward_eval)

        ])
        #self._summary_writer.add_summary(h_summaries, iteration)
        # self._summary_writer.add_summary(tf.summary.histogram('singular_values', s), iteration)
        self._summary_writer.add_summary(summ, iteration) 
        self._summary_writer.add_summary(summary, iteration)
