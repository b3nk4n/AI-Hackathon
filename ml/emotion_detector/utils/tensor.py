""" Tensorflow related utility functions. """
from __future__ import absolute_import, division, print_function

import tensorflow as tf


def get_num_trainable_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters