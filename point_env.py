# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym import spaces


class PointEnv(gym.Env):

    def __init__(self):  # Can set goal to test adaptation.
        # rng = np.random.RandomState(1)
        # goals = rng.uniform(-0.5, 0.5, size=(40, 2))
        # # print(goals)
        # self._goal = goals[env_config.worker_index - 1]
        # self._goal = np.array([0.5, 0.5])

        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(2, ), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
                                       shape=(2, ), dtype=np.float32)

    @property
    def reset_args_shape(self):
        return (2, )

    def sample_reset_args(self, rng, num):
        return rng.uniform(low=-0.5, high=0.5, size=(num, 2))

    def reset(self, reset_args=None):
        goal = reset_args
        if goal is not None:
            self._goal = goal

        self._state = (0, 0)
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        lb = self.action_space.low
        ub = self.action_space.high
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        self._state = self._state + scaled_action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        next_observation = np.clip(next_observation,
                                   self.observation_space.low,
                                   self.observation_space.high)
        return next_observation, reward, done, {"goal": self._goal}

    def render(self):
        print('current state:', self._state)
