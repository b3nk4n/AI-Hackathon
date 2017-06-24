# -!- coding: utf-8

from __future__ import absolute_import

import tensorflow as tf

from models.base import AbstractModel


class DexpressionClassifier(AbstractModel):
    def __init__(self, weight_decay, hyper_params):
        super(DexpressionClassifier, self).__init__(weight_decay)

        # some mappings to constant strings that are used to pass the hyper-paramers (HP) for the model
        # TODO: convert to property of the class so that can be access from outside while contructing the HP dict
        self.HP_CONVS = 'convs'
        self.HP_CONV_KERNEL_SIZE = 'kernel_size'
        self.HP_CONV_N_FILTERS = 'n_filters'
        self.HP_CONV_ACT_FN = 'act_fn'
        self.HP_CONV_REGL_FN = 'regl_fn'
        self.HP_MP = 'mp'
        self.HP_MP_POOL_SIZE = 'pool_size'
        self.HP_MP_STRIDES = self.HP_CONV_STRIDES = 'strides'

        self.hyper_params = hyper_params

    def inference(self, inputs, labels, is_training=True):
        # level 1
        conv_1 = self._conv2d(inputs, 'conv_1')
        pool_1 = self._max_pooling2d(conv_1, 'pool_1')
        # TODO: talk about this with Ben
        lrn_1 = None

        # level 2
        feat_ex_2 = self._feat_ex(lrn_1, '2')

        # level 3
        feat_ex_3 = self._feat_ex(feat_ex_2, '3')

        # classification layer
        # TODO
        classifier = None

        return classifier

    def _feat_ex(self, input, id):
        conv_a = self._conv2d(input, 'conv_{}a'.format(id))
        pool_a = self._max_pooling2d(input, 'pool_{}a'.format(id))

        conv_b = self._conv2d(conv_a, 'conv_{}b'.format(id))
        conv_c = self._conv2d(pool_a, 'conv_{}c'.format(id))

        concat = tf.concat([conv_b, conv_c], axis=0)

        pool_b = self._max_pooling2d(concat, 'pool_{}b'.format(id))

        return pool_b

    def _conv2d(self, inputs, conv_name):
        conv_hp = self.hyper_params[self.HP_CONVS][conv_name]
        # mandatory
        kernel_size = conv_hp[self.HP_CONV_KERNEL_SIZE]
        n_filters = conv_hp[self.HP_CONV_N_FILTERS]
        # optional
        strides = conv_hp.get(self.HP_CONV_STRIDES, (1, 1))
        activation_fn = conv_hp.get(self.HP_CONV_ACT_FN, tf.nn.relu)
        regularizer = conv_hp.get(self.HP_CONV_REGL_FN, None)

        # def: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d
        conv2d = tf.layers.conv2d(
            # custom
            inputs,
            n_filters,
            kernel_size,
            strides=strides,
            activation=activation_fn,
            kernel_initializer=tf.random_normal_initializer(),
            kernel_regularizer=regularizer,
            name=conv_name,
            # default
            padding='valid',
            data_format='channels_last',
            dilation_rate=(1, 1),
            use_bias=True,
            bias_initializer=tf.zeros_initializer(),
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            reuse=None
        )

        return conv2d

    def _max_pooling2d(self, inputs, max_pool_name):
        max_pool_hp = self.hyper_params[self.HP_MP][max_pool_name]
        # mandatory
        pool_size = max_pool_hp[self.HP_MP_POOL_SIZE]
        # optional
        strides = max_pool_hp.get(self.HP_MP_STRIDES, pool_size)

        reduced_weights = tf.layers.max_pooling2d(
            # custom
            inputs,
            pool_size,
            strides,
            name=max_pool_name,
            # defaults
            padding='valid',
            data_format='channels_last'
        )

        return reduced_weights




