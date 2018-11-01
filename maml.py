# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import ray
from ray.rllib.agents import Agent
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as ppo_default_config
from ray.rllib.utils import merge_dicts
from ray.rllib.evaluation.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.env.env_context import EnvContext
from ray.tune.trial import Resources

from maml_policy_graph import MAMLPolicyGraph
from maml_optimizer import MAMLOptimizer
from maml_policy_evaluator import MAMLPolicyEvaluator
from reset_wrapper import ResetArgsHolder


logger = logging.getLogger("ray.rllib.agents.maml.maml")

DEFAULT_CONFIG = merge_dicts(
    ppo_default_config,
    {
        "random_seed": 1,
        "inner_lr": 0.01,
        "outer_lr": 1e-3,
        "num_inner_updates": 3,
        "inner_grad_clip": 10.0,
        "num_tasks": 500,
        "clip_param": 0.2,
        "vf_share_layers": True,
        "use_gae": True,

        "gamma": 0.99,
        "lambda": 0.97,
        "horizon": 100,
        "kl_coeff": 0.0,
        "entropy_coeff": 0.0,
        "vf_loss_coeff": 0.05,
        "vf_clip_param": 20.0,

        "num_sgd_iter": 5,
        "sample_batch_size": 200,
        "batch_mode": "complete_episodes",
        "observation_filter": "NoFilter",
        "num_workers": 20,
        "num_envs_per_worker": 25,
        "tf_session_args": {
            "intra_op_parallelism_threads": 1,
            "inter_op_parallelism_threads": 1
        }
    }
)


class MAMLAgent(Agent):
    _agent_name = "MAML"
    _default_config = DEFAULT_CONFIG
    _policy_graph = MAMLPolicyGraph

    @classmethod
    def default_resource_request(cls, config):
        cf = merge_dicts(cls._default_config, config)
        return Resources(
            cpu=1,
            gpu=0,
            extra_cpu=cf["num_cpus_per_worker"] * cf["num_workers"],
            extra_gpu=cf["num_gpus_per_worker"] * cf["num_workers"])

    def make_local_evaluator(self, env_creator, policy_dict):
        return self._make_evaluator(
            MAMLPolicyEvaluator,
            env_creator,
            policy_dict,
            0,
            merge_dicts(
                self.config, {
                    "tf_session_args": {
                        "intra_op_parallelism_threads": None,
                        "inter_op_parallelism_threads": None
                    }
                }
            ))

    def make_remote_evaluators(self, env_creator, policy_dict, count,
                               remote_args):
        cls = MAMLPolicyEvaluator.as_remote(**remote_args).remote
        return [
            self._make_evaluator(cls, env_creator, policy_dict, i + 1,
                                 self.config) for i in range(count)
        ]

    def _init(self):
        self._validate_config()
        env = self.env_creator(EnvContext({"reset_args_holder": 100}, 0))

        self.reset_args_holder = ResetArgsHolder.remote(
            (self.config["num_workers"], ) + env.reset_args_shape)
        self.config["env_config"] = \
            {"reset_args_holder": self.reset_args_holder}

        self.rng = np.random.RandomState(self.config["random_seed"])
        self.all_reset_args = env.sample_reset_args(self.rng,
                                                    self.config["num_tasks"])

        observation_space = env.observation_space
        action_space = env.action_space
        policy_dict_local = {
            DEFAULT_POLICY_ID: (
                self._policy_graph,
                observation_space,
                action_space,
                {"mode": "local"})}
        policy_dict_remote = {
            DEFAULT_POLICY_ID: (
                self._policy_graph,
                observation_space,
                action_space,
                {"mode": "remote"})}

        self.local_evaluator = self.make_local_evaluator(
            self.env_creator, policy_dict_local)
        self.remote_evaluators = self.make_remote_evaluators(
            self.env_creator, policy_dict_remote, self.config["num_workers"], {
                "num_cpus": self.config["num_cpus_per_worker"],
                "num_gpus": self.config["num_gpus_per_worker"]})
        self.optimizer = MAMLOptimizer(
            self.local_evaluator, self.remote_evaluators, {
                "num_inner_updates": self.config["num_inner_updates"],
                "num_sgd_iter": self.config["num_sgd_iter"]})

    def _validate_config(self):
        # num_workers == meta_batch_size
        pass

    def _train(self):
        batch_reset_args_indices = \
            self.rng.choice(self.all_reset_args.shape[0],
                            size=self.config["num_workers"],
                            replace=False)
        batch_reset_args = self.all_reset_args[batch_reset_args_indices]
        ray.get(self.reset_args_holder.set.remote(batch_reset_args))

        fetches = self.optimizer.step()
        # if "kl" in fetches:
        #     raise NotImplementedError
        res = self.optimizer.collect_metrics()
        res.update(
            info=dict(fetches, **res.get("info", {})))
        return res

    def train(self):
        return Agent.__base__.train(self)


if __name__ == "__main__":
    import time
    import ray
    import numpy as np
    from ray.tune.registry import register_env
    from ray.rllib.models.catalog import ModelCatalog
    from ray.rllib.evaluation.metrics import summarize_episodes
    from ray.tune.logger import pretty_print
    from fcnet import FullyConnectedNetwork
    from point_env import PointEnv
    from reset_wrapper import ResetWrapper

    # logger = logging.getLogger("ray.rllib.agents.maml")
    # logger.setLevel(logging.DEBUG)

    ray.init()
    env_cls = PointEnv
    register_env(env_cls.__name__,
                 lambda env_config: ResetWrapper(env_cls(), env_config))
    # register_env("PointEnv", lambda env_config: PointEnv(env_config))
    ModelCatalog.register_custom_model("maml_mlp", FullyConnectedNetwork)

    config = {
        # "num_workers": 0,
        "model": {
            "custom_model": "maml_mlp",
            "fcnet_hiddens": [100, 100],
            "fcnet_activation": "tanh",
            "custom_options": {"vf_share_layers": True},
            # "squash_to_range": True,
            # "free_log_std": True
        }
    }

    agent = MAMLAgent(config=config, env=env_cls.__name__)
    evaluator = agent.local_evaluator
    policy = evaluator.policy_map[DEFAULT_POLICY_ID]
    optimizer = agent.optimizer

    for i in range(10):
        st = time.time()
        logger.info(f"\n{i}")
        res = agent.train()
        logger.info(f'\n{pretty_print(res["inner_update_metrics"])}')

    # only perform inner update in the local evaluator
    # policy.clear_grad_buffer()
    # def func():
    #     grads, infos, samples = evaluator._inner_update_once()
    #     policy.update_grad_buffer(grads)
    #     episodes = evaluator.sampler.get_metrics()
    #     logger.info(
    #         f'\n{pretty_print(summarize_episodes(episodes, episodes))}')
    #     logger.info(f"\n{pretty_print(infos)}")
    #     return grads, samples
    # for i in range(1000):
    #     print(i)
    #     grads, samples = func()
