# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
import ray
# from ray.experimental import named_actors


class ResetWrapper(gym.Wrapper):
    def __init__(self, env, env_config):
        assert not isinstance(env, self.__class__)
        gym.Wrapper.__init__(self, env)
        self.env_config = env_config
        self.reset_args_holder = env_config["reset_args_holder"]
        # set the following attribute in MAMLPolicyEvaluator.reset_sample
        self.with_reset_args = None

    @property
    def reset_args_shape(self):
        return self.env.reset_args_shape

    def sample_reset_args(self, *args, **kwargs):
        return self.env.sample_reset_args(*args, **kwargs)

    def reset(self):
        # reset_args = ray.get(
        #     named_actors.get_actor("reset_args").get.remote())
        if self.with_reset_args:
            this_reset_args = self.reset_args
        else:
            reset_args = ray.get(self.reset_args_holder.get.remote())
            this_reset_args = reset_args[self.env_config.worker_index - 1]
            self.reset_args = this_reset_args
            self.with_reset_args = True
        return self.env.reset(this_reset_args)

    def step(self, action):
        return self.env.step(action)


@ray.remote
class ResetArgsHolder(object):
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.args = np.zeros(shape)

    def get(self):
        return self.args

    def set(self, args):
        assert args.shape == self.shape
        self.args = args
