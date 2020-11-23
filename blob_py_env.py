from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1

class BlobEnv(py_environment.PyEnvironment):

  SIZE = 10
  RETURN_IMAGES = True
  MOVE_PENALTY = 1
  ENEMY_PENALTY = 300
  FOOD_REWARD = 25
  OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
  ACTION_SPACE_SIZE = 9
  PLAYER_N = 1  # player key in dict
  FOOD_N = 2  # food key in dict
  ENEMY_N = 3  # enemy key in dict
  # the dict! (colors)
  d = {1: (255, 175, 0),
        2: (0, 255, 0),
        3: (0, 0, 255)}

  def __init__(self):
    self.SIZE = 10
    #self._time_step_spec = tf.TensorSpec([1],dtype=np.uint8)
    self._state = 0
    self._episode_ended = False

  def time_step_spec(self):
    return ts.time_step_spec(self.observation_spec())

  def observation_spec(self):
    return BoundedArraySpec(
        shape=(1,10,10,3), dtype=np.uint8, minimum=0, maximum=255, name='observation')

  def action_spec(self):
    return BoundedArraySpec(
        shape=(1), dtype=np.int32, minimum=0, maximum=8, name='action')

  def reset(self):
    """Returns the current `TimeStep` after resetting the Environment."""
    return self._reset()

  def step(self, action):
    """Applies the action and returns the new `TimeStep`."""
    return self._step(action)

  def _reset(self):
    self._state = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
    self._episode_ended = False
    return ts.restart(self._state)

  def _step(self, action):

    self.episode_step += 1
    self.player.action(action)
    info = None

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    #Check action 
    if action < 0 or action > 8:
      raise ValueError('`action` should be 0 or 1.')

    new_observation = self.get_obs_array()
    if self.player == self.enemy:
        reward = -self.ENEMY_PENALTY
    elif self.player == self.food:
        reward = self.FOOD_REWARD
    else:
        reward = -self.MOVE_PENALTY

    ## Handle Terminal State
    if self._episode_ended:
      return ts.termination(np.array([self._state], dtype=np.uint8), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.uint8), reward=0.0, discount=1.0)
