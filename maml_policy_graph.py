# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf

from tensorflow.python.util import nest
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
# from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import _global_registry, RLLIB_MODEL

from base_maml_policy_graph import BaseMAMLPolicyGraph
from losses import A3CLoss, PPOLoss

logger = logging.getLogger("ray.rllib.agents.maml.maml_policy_graph")


class MAMLPolicyGraph(PPOPolicyGraph, BaseMAMLPolicyGraph):
    def __init__(self,
                 observation_space,
                 action_space,
                 config):
        # config = dict(ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG, **config)
        self.sess = tf.get_default_session()
        self.action_space = action_space
        self.config = config
        self.kl_coeff_val = self.config["kl_coeff"]
        self.kl_target = self.config["kl_target"]
        self.inner_lr = self.config["inner_lr"]
        self.outer_lr = self.config["outer_lr"]
        self.mode = self.config["mode"]
        assert self.mode in ["local", "remote"]
        assert self.kl_coeff_val == 0.0

        dist_cls, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"])

        with tf.name_scope("inputs"):
            obs_ph = tf.placeholder(
                tf.float32,
                shape=(None, ) + observation_space.shape,
                name="obs")
            adv_ph = tf.placeholder(
                tf.float32, shape=(None, ), name="advantages")
            act_ph = ModelCatalog.get_action_placeholder(action_space)
            logits_ph = tf.placeholder(
                tf.float32, shape=(None, logit_dim), name="logits")
            vf_preds_ph = tf.placeholder(
                tf.float32, shape=(None, ), name="vf_preds")
            value_targets_ph = tf.placeholder(
                tf.float32, shape=(None, ), name="value_targets")
            prev_actions_ph = ModelCatalog.get_action_placeholder(action_space)
            prev_rewards_ph = tf.placeholder(
                tf.float32, shape=(None, ), name="prev_rewards")
            existing_state_in = None
            existing_seq_lens = None

        self.observations = obs_ph

        self.a3c_loss_in = [
            ("obs", obs_ph),
            ("advantages", adv_ph),
            ("actions", act_ph),
            ("value_targets", value_targets_ph),
            ("vf_preds", vf_preds_ph),
            ("prev_actions", prev_actions_ph),
            ("prev_rewards", prev_rewards_ph)
        ]
        # self.a3c_loss_in = [
        #     ("obs", obs_ph),
        #     ("advantages", adv_ph),
        #     ("actions", act_ph),
        #     ("prev_actions", prev_actions_ph),
        #     ("prev_rewards", prev_rewards_ph)]
        self.ppo_loss_in = list(self.a3c_loss_in) \
            + [("logits", logits_ph)]

        assert self.config["model"]["custom_model"]
        logger.info(
            f'Using custom model {self.config["model"]["custom_model"]}')
        model_cls = _global_registry.get(RLLIB_MODEL,
                                         self.config["model"]["custom_model"])
        new_variables, grad_placeholders, custom_variables, dummy_variables = \
            model_cls.prepare(observation_space,
                              (logit_dim // 2
                               if self.config["model"]["free_log_std"]
                               else logit_dim),
                              self.config["model"],
                              func=lambda x, y: x - self.inner_lr * y)
        self._new_variables = new_variables
        self._grad_placeholders = grad_placeholders
        self._custom_variables = custom_variables
        self._dummy_variables = dummy_variables
        self._inner_variables = nest.flatten(custom_variables)
        # for Meta-SGD, `custom_variables` and `adaptive learning rates`
        self._outer_variables = nest.flatten(custom_variables)

        self._variables = {var.op.name: var for var in self._outer_variables}

        self._grad_phs_loss_inputs = []
        for i in range(len(grad_placeholders)):
            for key, ph in grad_placeholders[i].items():
                self._grad_phs_loss_inputs.append(
                    (custom_variables[i][key].op.name, ph))
        self._grad_phs_loss_input_dict = dict(self._grad_phs_loss_inputs)

        self.model = model_cls(
            {
                "obs": obs_ph,
                "prev_actions": prev_actions_ph,
                "prev_rewards": prev_actions_ph
            },
            observation_space,
            logit_dim,
            self.config["model"],
            state_in=existing_state_in,
            seq_lens=existing_seq_lens,
            custom_params=new_variables)

        self.logits = self.model.outputs
        with tf.name_scope("sampler"):
            curr_action_dist = dist_cls(self.logits)
            self.sampler = curr_action_dist.sample()

        assert self.config["use_gae"] and self.config["vf_share_layers"]
        self.value_function = self.model.value_function()

        if self.model.state_in:
            raise NotImplementedError
        else:
            mask = None

        with tf.name_scope("a3c_loss"):
            self.a3c_loss_obj = A3CLoss(
                action_dist=curr_action_dist,
                actions=act_ph,
                advantages=adv_ph,
                value_targets=value_targets_ph,
                vf_preds=vf_preds_ph,
                value_function=self.value_function,
                vf_loss_coeff=self.config["vf_loss_coeff"],
                entropy_coeff=self.config["entropy_coeff"],
                vf_clip_param=self.config["vf_clip_param"])
            # self.a3c_loss_obj = PGLoss(
            #     curr_action_dist, act_ph, adv_ph)
        with tf.name_scope("ppo_loss"):                  # write own PPO loss, boolean_mask -> dynamic_partition
            self.ppo_loss_obj = PPOLoss(
                action_dist=curr_action_dist,
                action_space=action_space,
                logits=logits_ph,
                actions=act_ph,
                advantages=adv_ph,
                value_targets=value_targets_ph,
                vf_preds=vf_preds_ph,
                value_function=self.value_function,
                valid_mask=mask,
                kl_coeff=self.kl_coeff_val,
                clip_param=self.config["clip_param"],
                vf_clip_param=self.config["vf_clip_param"],
                vf_loss_coeff=self.config["vf_loss_coeff"],
                entropy_coeff=self.config["entropy_coeff"],
                use_gae=self.config["use_gae"])

        BaseMAMLPolicyGraph.__init__(
            self,
            observation_space,
            action_space,
            self.sess,
            obs_input=obs_ph,
            action_sampler=self.sampler,
            inner_loss=self.a3c_loss_obj.loss,
            inner_loss_inputs=self.a3c_loss_in,
            outer_loss=self.ppo_loss_obj.loss,
            outer_loss_inputs=self.ppo_loss_in,
            state_inputs=self.model.state_in,
            state_outputs=self.model.state_out,
            prev_action_input=prev_actions_ph,
            prev_reward_input=prev_rewards_ph,
            seq_lens=self.model.seq_lens,
            max_seq_len=self.config["model"]["max_seq_len"])

        self.a3c_stats_fetches = {
            "total_loss": self.a3c_loss_obj.loss,
            "policy_loss": self.a3c_loss_obj.mean_policy_loss,
            "vf_loss": self.a3c_loss_obj.mean_vf_loss,
            "entropy": self.a3c_loss_obj.mean_entropy
        }
        self.ppo_stats_fetches = {
            "total_loss": self.ppo_loss_obj.loss,
            "policy_Loss": self.ppo_loss_obj.mean_policy_loss,
            "vf_loss": self.ppo_loss_obj.mean_vf_loss,
            "entropy": self.ppo_loss_obj.mean_entropy,
            "kl": self.ppo_loss_obj.mean_kl
        }

        self.sess.run(tf.global_variables_initializer())
        # self.clear_grad_buffer()

    def clear_grad_buffer(self):
        self._grad_buffer = {
            name: np.zeros(ph.shape.as_list(),
                           dtype=ph.dtype.as_numpy_dtype)
            for name, ph in self._grad_phs_loss_input_dict.items()}

    def update_grad_buffer(self, grad_values):
        for key, grad in grad_values.items():
            self._grad_buffer[key] += grad

    def extra_compute_action_feed_dict(self):
        feed_dict = {
            self._grad_phs_loss_input_dict[name]: self._grad_buffer[name]
            for name in self._grad_phs_loss_input_dict}
        return feed_dict

    def extra_compute_grad_feed_dict(self):
        feed_dict = self.extra_compute_action_feed_dict()
        return feed_dict

    def extra_compute_grad_fetches(self):
        return self.stats_fetches

    def _get_inner_grads(self):
        inner_grads = \
            tf.gradients(self._inner_loss, self._inner_variables,
                         name="inner_gradients")
        clipped_inner_grads, _ = \
            tf.clip_by_global_norm(inner_grads, self.config["inner_grad_clip"])
        return {
            v.op.name: g
            for v, g in zip(self._inner_variables, clipped_inner_grads)
            if g is not None}

    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.config["outer_lr"])

    # def postprocess_trajectory(self, sample_batch, other_agent_batches=None):
    #     return compute_advantages(
    #         sample_batch, 0.0, self.config["gamma"], use_gae=False)


if __name__ == "__main__":
    import gym
    import ray
    from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
    from ray.rllib.evaluation.policy_evaluator import PolicyEvaluator
    from ray.tune.logger import pretty_print
    from fcnet import FullyConnectedNetwork

    # ray.init()
    ModelCatalog.register_custom_model("maml_mlp", FullyConnectedNetwork)

    config = {
        "inner_lr": 0.5,
        "outer_lr": 0.0001,
        "use_gae": True,
        "vf_share_layers": True,
        "horizon": 200,
        "batch_mode": "complete_episodes",
        "observation_filter": "NoFilter",
        "model": {
            "custom_model": "maml_mlp",
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
            "max_seq_len": 20,
            "custom_options": {"vf_share_layers": True}
        }
    }
    config = dict(DEFAULT_CONFIG, **config)
    print(pretty_print(config))

    sess = tf.InteractiveSession()

    def env_creator(config):
        return gym.make("CartPole-v1")

    evaluator = PolicyEvaluator(
        env_creator,
        MAMLPolicyGraph,
        batch_steps=config["sample_batch_size"],
        batch_mode=config["batch_mode"],
        episode_horizon=config["horizon"],
        preprocessor_pref=config["preprocessor_pref"],
        sample_async=config["sample_async"],
        compress_observations=config["compress_observations"],
        num_envs=config["num_envs_per_worker"],
        observation_filter=config["observation_filter"],
        clip_rewards=config["clip_rewards"],
        env_config=config["env_config"],
        model_config=config["model"],
        policy_config=config,
        worker_index=0,
        monitor_path=self.logdir if config["monitor"] else None,
        log_level=config["log_level"])
    policy = evaluator.policy_map["default"]
    batch = evaluator.sample()
    grads, infos = policy.compute_inner_gradients(batch)

    # observation_space = env.observation_space
    # action_space = env.action_space
    # policy_graph = MAMLPolicyGraph(observation_space, action_space, config)
    # graph = tf.get_default_graph()
    # writer = tf.summary.FileWriter(logdir="./summary", graph=graph)
    writer = tf.summary.FileWriter(logdir="./summary", graph=evaluator.tf_sess.graph)
    writer.flush()
