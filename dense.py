# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras import models as keras_models
from ray.rllib.models.misc import normc_initializer


class KerasFCModel(keras_models.Model):
    @classmethod
    def get_dummy_variables(cls, all_units, vf_share_layers=False, **kwargs):
        variables = []
        dummy_instance = cls(dummy=True)
        with tf.name_scope(dummy_instance._name_scope()):
            for i, (size_in, size_out) in \
                    enumerate(zip(all_units, all_units[1:])):
                name = f"fc_{i}"
                variables.append(
                    Dense.get_dummy_variables(size_in, size_out,
                                              name=name, **kwargs))
            if vf_share_layers:
                name = f"fc_vf"
                variables.append(
                    Dense.get_dummy_variables(size_in, 1,
                                              name=name, **kwargs))
        # tricky to remove the name count of the dummy instance
        graph = tf.get_default_graph()
        backend.PER_GRAPH_LAYER_NAME_UIDS[graph][
            ("", dummy_instance.name)] -= 1
        return variables

    @staticmethod
    def filter_dummy_variables(variables):
        assert isinstance(variables, list)
        filtered_variables = []
        for var in variables:
            tmp = {}
            for key, value in var.items():
                if key in ["kernel", "bias"]:
                    tmp[key] = value
            filtered_variables.append(tmp)
        return filtered_variables

    def __init__(self, layer_units=None, activation=None,
                 custom_params=None, vf_share_layers=False, dummy=False):
        """
            layer_units: list, a list of the number of units of all layers
                except the input layer
        """
        keras_models.Model.__init__(self)
        if dummy:
            return
        assert layer_units is not None and activation is not None

        def _get_initializer(i, n):
            if i < len(n) - 1:
                return normc_initializer(1.0)
            else:
                return normc_initializer(0.01)

        if not custom_params:
            for i, size in enumerate(layer_units):
                name = f"fc_{i}"
                layer = Dense(
                    size,
                    activation=(
                        activation if i < len(layer_units) - 1 else None),
                    kernel_initializer=_get_initializer(i, layer_units),
                    name=name)
                setattr(self, name, layer)
            if vf_share_layers:
                name = f"fc_vf"
                layer = Dense(
                    1,
                    activation=None,
                    kernel_initializer=normc_initializer(1.0),
                    name=name)
                setattr(self, name, layer)
        else:
            if vf_share_layers:
                assert len(layer_units) == len(custom_params) - 1
            else:
                assert len(layer_units) == len(custom_params)
            for i, size in enumerate(layer_units):
                name = f"fc_{i}"
                layer = Dense(
                    custom_params=custom_params[i],
                    activation=(
                        activation if i < len(layer_units) - 1 else None),
                    name=name)
                setattr(self, name, layer)
            if vf_share_layers:
                name = f"fc_vf"
                layer = Dense(
                    custom_params=custom_params[-1],
                    activation=None,
                    name=name)
                setattr(self, name, layer)
        self._vf_share_layers = vf_share_layers

    def call(self, inputs):
        last_inputs = inputs
        last_shared_index = -2 if self._vf_share_layers else -1
        for layer in self.layers[:last_shared_index]:
            last_inputs = layer(last_inputs)
        output = self.layers[last_shared_index](last_inputs)
        if self._vf_share_layers:
            value_function = self.layers[-1](last_inputs)
            return output, value_function
        else:
            return output


class Dense(keras_layers.Dense):
    @classmethod
    def get_dummy_variables(cls, input_units, output_units, **kwargs):
        dummy_instance = cls(output_units, **kwargs)
        with tf.name_scope(dummy_instance._name_scope()):
            dummy_instance.build(input_shape=[None, input_units])
        # maybe just return useful infos, e.g., regularization loss
        # instead of the dummy instance
        variables = {
            "dummy": dummy_instance,
            "kernel": dummy_instance.kernel}
        if dummy_instance.use_bias:
            variables["bias"] = dummy_instance.bias
        return variables

    def __init__(self, units=None, custom_params=None, **kwargs):
        assert (units is not None) != (custom_params is not None)
        units = units or custom_params["kernel"].shape.as_list()[-1]

        keras_layers.Dense.__init__(self, units, **kwargs)

        if custom_params:
            assert isinstance(custom_params, dict)
            param_names = list(custom_params.keys())
            assert "kernel" in param_names
            if self.use_bias:
                assert "bias" in param_names
        self.custom_params = custom_params

    def build(self, input_shape):
        if self.custom_params:
            self.kernel = self.custom_params["kernel"]
            if self.use_bias:
                self.bias = self.custom_params["bias"]
            self.built = True
        else:
            keras_layers.Dense.build(self, input_shape)
