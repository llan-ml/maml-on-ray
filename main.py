# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune.registry import register_trainable, register_env
from ray.tune import run_experiments, grid_search
from ray.rllib.models.catalog import ModelCatalog

from maml import MAMLAgent
from point_env import PointEnv
from reset_wrapper import ResetWrapper
from fcnet import FullyConnectedNetwork


register_trainable("MAML", MAMLAgent)
env_cls = PointEnv
register_env(env_cls.__name__,
             lambda env_config: ResetWrapper(env_cls(), env_config))
ModelCatalog.register_custom_model("maml_mlp", FullyConnectedNetwork)

# ray.init()
ray.init(redis_address="localhost:32222")

config = {
    "random_seed": grid_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    "inner_lr": grid_search([0.01]),
    "inner_grad_clip": grid_search([10.0, 20.0, 30.0, 40.0]),
    "clip_param": grid_search([0.1, 0.2, 0.3]),
    "vf_loss_coeff": grid_search([0.01, 0.02, 0.05, 0.1, 0.2]),
    "vf_clip_param": grid_search([5.0, 10.0, 15.0, 20.0]),
    "model": {
        "custom_model": "maml_mlp",
        "fcnet_hiddens": [100, 100],
        "fcnet_activation": "tanh",
        "custom_options": {"vf_share_layers": True}
    }
}

run_experiments({
    "maml_point": {
        "run": "MAML",
        "env": env_cls.__name__,
        "stop": {"training_iteration": 500},
        "config": config,
        "local_dir": "/ray_results"
    }
})
