# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import ray
import tensorflow as tf
from ray.tune.registry import register_env
from ray.rllib.agents.pg.pg import PGAgent
from ray.rllib.agents.pg.pg_policy_graph import PGPolicyGraph
from ray.rllib.agents.a3c.a2c import A2CAgent
from ray.rllib.agents.a3c.a3c_tf_policy_graph import A3CPolicyGraph
from ray.rllib.agents.ppo.ppo import PPOAgent
from ray.rllib.optimizers.sync_samples_optimizer import logger as opt_logger
from ray.tune.logger import pretty_print
from point_env import PointEnv

opt_logger.setLevel(logging.DEBUG)


ray.init()
register_env("PointEnv", lambda env_config: PointEnv())

config = {
    "gamma": 0.99,
    "lambda": 0.97,
    "vf_clip_param": 5.0,
    "lr": 5e-5,
    "num_sgd_iter": 30,
    "sample_batch_size": 200,
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
    "num_workers": 5,
    "num_envs_per_worker": 20,
    "horizon": 100,
    "vf_share_layers": True,
    "clip_param": 0.2,
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "tanh",
        # "free_log_std": True
    },
    "vf_loss_coeff": 0.02,
    "entropy_coeff": 0.0,
    "kl_coeff": 0.0,
    "simple_optimizer": True
}


# class NewPolicyGraph(A3CPolicyGraph):
#     def optimizer(self):
#         # return tf.train.GradientDescentOptimizer(0.001)
#         return tf.train.AdamOptimizer()


# def _train(self):
#     prev_steps = self.optimizer.num_steps_sampled
#     # start = time.time()
#     # while time.time() - start < self.config["min_iter_time_s"]:
#     self.optimizer.step()
#     result = self.optimizer.collect_metrics()
#     result.update(timesteps_this_iter=self.optimizer.num_steps_sampled -
#                   prev_steps)
#     return result


# PGAgent._policy_graph = NewPolicyGraph
# A2CAgent._policy_graph = NewPolicyGraph
# A2CAgent._train = _train

# agent = PGAgent(config=config, env="PointEnv")
# agent = A2CAgent(config=config, env="PointEnv")
agent = PPOAgent(config=config, env="PointEnv")
# res = agent.train()
for i in range(1000):
    print(i)
    res = agent.train()
    print(pretty_print({key: value for key, value in res.items()
                        if key.startswith("episode")}))
