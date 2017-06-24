# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging

import tensorflow as tf

from models.base import AbstractModel


logger = logging.getLogger(__name__)


class DexpressionNet(AbstractModel):
    @property
    def HP_CONV(self):
        return 'convs'

    @property
    def HP_ALL_CONVS(self):
        def convs_from_feat_ex(l):
            return ['conv_{}{}'.format(l, p) for p in ['a', 'b', 'c']]

        return ['conv_1'] + convs_from_feat_ex('2') + convs_from_feat_ex('3')

    @property
    def HP_CONV_KERNEL_SIZE(self):
        return 'kernel_size'

    @property
    def HP_CONV_N_FILTERS(self):
        return 'n_filters'

    @property
    def HP_CONV_ACTIVATION_FN(self):
        return 'act_fn'

    @property
    def HP_MP(self):
        return 'mp'

    @property
    def HP_ALL_MPS(self):
        def mps_from_feat_ex(l):
            return ['pool_{}{}'.format(l, p) for p in ['a', 'b']]

        return ['pool_1'] + mps_from_feat_ex('2') + mps_from_feat_ex('3')

    @property
    def HP_MP_POOL_SIZE(self):
        return 'pool_size'

    @property
    def HP_FC(self):
        return 'fc'

    @property
    def HP_FC_N_OUTPUTS(self):
        return 'n_outputs'

    @property
    def HP_FC_DROPOUT_RATE(self):
        return 'dropout_rate'

    @property
    def __REGL_FN(self):
        return 'regl_fn'

    @property
    def HP_FC_REGL_FN(self):
        return self.__REGL_FN

    @property
    def HP_CONV_REGL_FN(self):
        return self.__REGL_FN

    @property
    def __HP_STRIDES(self):
        return 'strides'

    @property
    def HP_MP_STRIDES(self):
        return self.__HP_STRIDES

    @property
    def HP_CONV_STRIDES(self):
        return self.__HP_STRIDES

    @property
    def __HP_PADDING_TYPE(self):
        return 'padding_type'

    @property
    def HP_CONV_PADDING_TYPE(self):
        return self.__HP_PADDING_TYPE

    @property
    def HP_MP_PADDING_TYPE(self):
        return self.__HP_PADDING_TYPE

    def __init__(self, weight_decay, hyper_params):
        super(DexpressionNet, self).__init__(weight_decay)

        # some mappings to constant strings that are used to pass the hyper-paramers (HP) for the model
        # TODO: convert to property of the class so that can be access from outside while contructing the HP dict
        # related to convolution layers

        # self.HP_CONV_KERNEL_SIZE = 'kernel_size'
        # self.HP_CONV_N_FILTERS = 'n_filters'
        # self.HP_CONV_ACT_FN = 'act_fn'
        # # related to max-pooling ops (layers)
        # self.HP_MP = 'mp'
        # self.HP_MP_POOL_SIZE = 'pool_size'
        # self.HP_CONV_REGL_FN = 'regl_fn'
        # # some shared names
        # self.HP_FC_REGL_FN = self.HP_CONV_REGL_FN = 'regl_fn'
        # self.HP_MP_STRIDES = self.HP_CONV_STRIDES = 'strides'

        self.hyper_params = hyper_params

        # will bet set in the inference()
        self._classifier = None
        # will be set in loss()
        self._loss = None

    def __feat_ex(self, input, id):
        conv_a = self.__conv2d(input, 'conv_{}a'.format(id))
        pool_a = self.__max_pooling2d(input, 'pool_{}a'.format(id))

        conv_b = self.__conv2d(conv_a, 'conv_{}b'.format(id))
        conv_c = self.__conv2d(pool_a, 'conv_{}c'.format(id))

        concat = tf.concat([conv_b, conv_c], axis=0)

        pool_b = self.__max_pooling2d(concat, 'pool_{}b'.format(id))

        return pool_b

    def __conv2d(self, inputs, conv_name):
        conv_hp = self.hyper_params[self.HP_CONV][conv_name]
        # mandatory
        kernel_size = conv_hp[self.HP_CONV_KERNEL_SIZE]
        n_filters = conv_hp[self.HP_CONV_N_FILTERS]
        # optional
        strides = conv_hp.get(self.HP_CONV_STRIDES, (1, 1))
        activation_fn = conv_hp.get(self.HP_CONV_ACTIVATION_FN, tf.nn.relu)
        regularizer = conv_hp.get(self.HP_CONV_REGL_FN, tf.contrib.layers.l2_regularizer(self._weight_decay))
        padding_type = conv_hp.get(self.HP_CONV_PADDING_TYPE, 'same')

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
            padding=padding_type,
            # default
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

    def __max_pooling2d(self, inputs, max_pool_name):
        max_pool_hp = self.hyper_params[self.HP_MP][max_pool_name]
        # mandatory
        pool_size = max_pool_hp[self.HP_MP_POOL_SIZE]
        # optional
        strides = max_pool_hp.get(self.HP_MP_STRIDES, pool_size)
        padding_type = max_pool_hp.get(self.HP_MP_PADDING_TYPE, 'same')

        reduced_weights = tf.layers.max_pooling2d(
            # custom
            inputs,
            pool_size,
            strides,
            name=max_pool_name,
            padding=padding_type,
            # defaults
            data_format='channels_last'
        )
        return reduced_weights

    def __fully_connected(self, inputs, fc_name, n_outs=None):
        fc_hp = self.hyper_params[self.HP_FC][fc_name]
        n_outs = n_outs if n_outs else fc_hp[self.HP_FC_N_OUTPUTS]

        # since 'fc_1' is optional
        if not n_outs:
            return None, True

        # optional
        regularizer = fc_hp.get(self.HP_FC_REGL_FN, None)

        dense_layer = tf.layers.dense(
            # custom
            inputs,
            n_outs,
            kernel_regularizer=regularizer,
            # default
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name=None,
            reuse=None
        )
        return dense_layer, False

    def __batch_normalization(self, inputs, is_training, name):
        batch_norm = tf.layers.batch_normalization(
            # custom
            inputs,
            name=name,
            training=is_training,
            # default
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            beta_regularizer=None,
            gamma_regularizer=None,
            trainable=True,
            reuse=None,
            renorm=False,
            renorm_clipping=None,
            renorm_momentum=0.99
        )
        return batch_norm

    def __dropout(self, inputs, after_layer_name):
        after_layer_hp = self.hyper_params[self.HP_FC][after_layer_name]
        # mandatory
        dropout_rate = after_layer_hp[self.HP_FC_DROPOUT_RATE]

        out = tf.layers.dropout(
            # custom
            inputs,
            rate=dropout_rate,
            name="dropout_{}".format(after_layer_name),
            # default
            noise_shape=None,
            seed=None,
            training=False
        )
        return out

    def inference(self, inputs, labels, is_training=True):
        # level 1
        conv_1 = self.__conv2d(inputs, 'conv_1')
        pool_1 = self.__max_pooling2d(conv_1, 'pool_1')

        # we are using batch normalisation instead of lrn
        lrn_1 = self.__batch_normalization(pool_1, is_training, 'lrn_1')

        # level 2
        feat_ex_2 = self.__feat_ex(lrn_1, '2')

        # level 3
        feat_ex_3 = self.__feat_ex(feat_ex_2, '3')
        flattened_feat_ex_3 = tf.reshape(feat_ex_3, [-1])

        # optional fully connected which is not in the paper
        fc_1, skip_fc_1 = self.__fully_connected(flattened_feat_ex_3, 'fc_1')

        if skip_fc_1:
            fc_1 = flattened_feat_ex_3
        else:
            logging.warning("You have added an optional fully connected layer to the network which is not " +
                            "a part of the original DeXepression network. A dropout will also be applied" +
                            "after the fully connected layer.")
            fc_1 = self.__dropout(fc_1, 'fc_1')

        # classification layer
        self._classifier = self.__fully_connected(fc_1, 'out', labels.get_shape().as_list()[-1])

        return self._classifier

    def loss(self, predictions, labels):
        # note tf.nn.softmax_cross_entropy_with_logits expects pred to be unscaled,
        # since it performs a softmax on logits internally for efficiency. Otherwise it is same as -
        #  -(y * log(softmax(pred)))
        # note the 'minus' at the front of the above equation
        # do not call this op with the output of softmax, as it will produce incorrect results.
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels))
        return self._loss

    def validation(self, predictions, labels):
        # Evaluating the model
        true_positives = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        avg_accuracy = tf.reduce_mean(tf.cast(true_positives, tf.float32))
        return avg_accuracy

        # TODO:
#   1. hyperparams optimization
#   2. adding update to default or custom collection? - side effects on performance