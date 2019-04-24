from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math



import gym
import gym_gridworlds
import numpy as np
import tensorflow as tf
import dopamine.discrete_domains.gym_lib
import gin.tf

gin.constant('gridworld_lib.BIG_OBSERVATION_SHAPE', (4, 1))
gin.constant('gridworld_lib.SMALL_OBSERVATION_SHAPE', (4, 4))
gin.constant('gridworld_lib.OBSERVATION_DTYPE', tf.uint8)

slim = tf.contrib.slim


@gin.configurable
def create_grid_environment(environment_name=None, version='v0'):
  """Wraps a Gym environment with some basic preprocessing.

  Args:
    environment_name: str, the name of the environment to run.
    version: str, version of the environment to run.

  Returns:
    A Gym environment with some standard preprocessing.
  """
  assert environment_name is not None
  full_game_name = '{}-{}'.format(environment_name, version)
  env = gym.make(full_game_name)

  # Wrap the returned environment in a class which conforms to the API expected
  # by Dopamine.
  env = GridworldPreprocessing(env)
  return env

@gin.configurable
def _basic_grid_network(num_actions, state,
                                   num_atoms=None, bottleneck=10, max_dim=10):
  """Builds a basic network for discrete domains, rescaling inputs to [-1, 1].

  Args:
    min_vals: float, minimum attainable values (must be same shape as `state`).
    max_vals: float, maximum attainable values (must be same shape as `state`).
    num_actions: int, number of actions.
    state: `tf.Tensor`, the state input.
    num_atoms: int or None, if None will construct a DQN-style network,
      otherwise will construct a Rainbow-style network.

  Returns:
    The Q-values for DQN-style agents or logits for Rainbow-style agents.
  """
  net = tf.cast(state, tf.float32)
  net = slim.flatten(net)
  net /= max_dim
  print("Building basic grid network")
  print("bottleneck: ", bottleneck)
  net = tf.cast(state, tf.float32)
  net = slim.flatten(net)
  net = slim.fully_connected(net, 256)
  net = slim.fully_connected(net, 256)
  net = slim.fully_connected(net, bottleneck)
  # net = tf.Print(net, [tf.boolean_mask(net, mask)], summarize=10)
  
  if num_atoms is None:
    # We are constructing a DQN-style network.
    return slim.fully_connected(net, num_actions, activation_fn=None), net
  else:
    # We are constructing a rainbow-style network.
    return slim.fully_connected(net, num_actions * num_atoms,
                                activation_fn=None), net




@gin.configurable
def gridworld_dqn_network(num_actions, network_type, state, bottleneck=10):
  """Builds the deep network used to compute the agent's Q-values.

  It rescales the input features to a range that yields improved performance.

  Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  q_values, features = _basic_grid_network(
     num_actions, state, bottleneck=bottleneck)
  return network_type(q_values, features)


@gin.configurable
def gridworld_feature_network(state, bottleneck=10):
  features = _basic_grid_features(state, bottleneck)
  return features


@gin.configurable
def gridworld_rainbow_network(num_actions, num_atoms, support,
 network_type, state, bottleneck=10):
  net, features = _basic_grid_network(
    num_actions, state,
    num_atoms=num_atoms, bottleneck=bottleneck)
  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = tf.contrib.layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities, features)

@gin.configurable
class GridworldPreprocessing(object):
  """A Wrapper class around Gym environments."""

  def __init__(self, environment):
    self.environment = environment
    self.game_over = False

  @property
  def observation_space(self):
    return self.environment.observation_space

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def reset(self):
    return self.environment.reset()

  def step(self, action):
    observation, reward, game_over, info = self.environment.step(action)
    self.game_over = game_over
    return observation, reward, game_over, info
  
  @property
  def height(self):
    return self.environment.height

  @property
  def width(self):
    return self.environment.width

