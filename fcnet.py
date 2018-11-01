# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.util import nest
from ray.rllib.models import model as rllib_model
from ray.rllib.models.misc import get_activation_fn

from dense import KerasFCModel


class Model(rllib_model.Model):
    def __init__(self,
                 input_dict,
                 obs_space,
                 num_outputs,
                 options,
                 state_in=None,
                 seq_lens=None,
                 custom_params=None):
        self.custom_params = custom_params
        rllib_model.Model.__init__(
            self,
            input_dict,
            obs_space,
            num_outputs,
            options,
            state_in,
            seq_lens)


class FullyConnectedNetwork(Model):
    KERAS_MODEL = KerasFCModel

    def _build_layers(self, inputs, num_outputs, options):
        hiddens = options.get("fcnet_hiddens")
        activation = get_activation_fn(options.get("fcnet_activation"))
        vf_share_layers = options.get("custom_options").get("vf_share_layers")

        model = self.KERAS_MODEL(
            layer_units=hiddens + [num_outputs],
            activation=activation,
            custom_params=self.custom_params,
            vf_share_layers=vf_share_layers)
        self.keras_model = model
        if vf_share_layers:
            output, value_function = model(inputs)
            self._value_function = tf.reshape(value_function, [-1])
        else:
            output = model(inputs)
        last_layer = model.layers[-1]
        return output, last_layer

    def value_function(self):
        return self._value_function

    @staticmethod
    def prepare(observation_space, output_dim, options, func=None):
        assert len(observation_space.shape) == 1
        all_units = observation_space.shape \
            + tuple(options["fcnet_hiddens"]) \
            + (output_dim, )
        vf_share_layers = options["custom_options"]["vf_share_layers"]

        with tf.name_scope("variables"):
            dummy_variables = FullyConnectedNetwork \
                .KERAS_MODEL.get_dummy_variables(all_units, vf_share_layers)
        custom_variables = FullyConnectedNetwork \
            .KERAS_MODEL.filter_dummy_variables(dummy_variables)
        with tf.name_scope("placeholders"):
            new_variables, placeholders = \
                build_placeholder_and_transform(custom_variables, func=func)
        return new_variables, placeholders, custom_variables, dummy_variables


def build_placeholder_and_transform(variables, func=None):
    if func is None:
        def func(x, y):
            return x - y if y is not None else None
        assert False

    import re

    def _get_name(x):
        return re.sub(".*variables/", "", x)

    flat_variables = nest.flatten(variables)
    assert all([isinstance(var, tf.Variable) for var in flat_variables])
    flat_placeholders = [
        tf.placeholder(var.dtype, var.shape, _get_name(var.op.name))
        for var in flat_variables]
    flat_new_variables = list(map(func, flat_variables, flat_placeholders))
    placeholders = nest.pack_sequence_as(variables, flat_placeholders)
    new_variables = nest.pack_sequence_as(variables, flat_new_variables)
    return new_variables, placeholders


if __name__ == "__main__":
    import os
    tf.set_random_seed(1)
    mode = 2

    x = tf.placeholder(tf.float32, [None, 10], "x")
    # x = keras_layers.Input(tensor=x)

    input_dict = {"obs": x, "prev_action": None, "prev_reward": None}
    obs_space = None
    num_outputs = 5
    options = {
        "fcnet_hiddens": [20],
        "fcnet_activation": "tanh",
        "free_log_std": False,
        "vf_share_layers": True}

    if mode == 1:
        model = FullyConnectedNetwork(
            input_dict=input_dict,
            obs_space=obs_space,
            num_outputs=num_outputs,
            options=options)
    elif mode == 2:
        new_variables, placeholders, custom_variables, dummy_variables = \
            FullyConnectedNetwork.prepare([10, 20, 5], vf_share_layers=True)
        model = FullyConnectedNetwork(
            input_dict,
            obs_space,
            num_outputs,
            options,
            custom_params=new_variables)

    init_op = tf.global_variables_initializer()

    graph = tf.get_default_graph()
    os.system("mkdir -p summary")
    writer = tf.summary.FileWriter(logdir="./summary", graph=graph)
    writer.flush()

    sess = tf.Session()
    sess.run(init_op)
    # print(sess.run(tf.trainable_variables()))
