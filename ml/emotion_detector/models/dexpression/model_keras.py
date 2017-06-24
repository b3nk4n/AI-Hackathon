# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging
import coloredlogs

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from models.base import AbstractModel

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


def create_model(image_size):
    model = Sequential()
    model.add(Conv2D(32, 3, 3, input_shape=(image_size, image_size, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('sigmoid'))
    return model


class DexpressionNet(AbstractModel):
    @staticmethod
    def conv_names_from_feat_ex(l):
        return ['conv_{}{}'.format(l, p) for p in ['a', 'b', 'c']]

    @staticmethod
    def mp_names_from_feat_ex(l):
        return ['pool_{}{}'.format(l, p) for p in ['a', 'b']]

    HP_CONV = 'convs'
    HP_CONV_KERNEL_SIZE = 'kernel_size'
    HP_CONV_N_FILTERS = 'n_filters'

    HP_MP = 'mp'
    HP_MP_POOL_SIZE = 'pool_size'

    HP_FC = 'fc'
    HP_FC_N_OUTPUTS = 'n_outputs'
    HP_FC_DROPOUT_RATE = 'dropout_rate'

    __ACTIVATION_FN = 'act_fn'
    HP_CONV_ACTIVATION_FN = __ACTIVATION_FN
    HP_FC_ACTIVATION_FN = __ACTIVATION_FN

    __REGL_FN = 'regl_fn'
    HP_FC_REGL_FN = __REGL_FN
    HP_CONV_REGL_FN = __REGL_FN

    __HP_STRIDES = 'strides'
    HP_CONV_STRIDES = __HP_STRIDES
    HP_MP_STRIDES = __HP_STRIDES

    __HP_PADDING_TYPE = 'padding_type'
    HP_CONV_PADDING_TYPE = __HP_PADDING_TYPE
    HP_MP_PADDING_TYPE = __HP_PADDING_TYPE

    def __init__(self, weight_decay, input_size, hyper_params):
        super(DexpressionNet, self).__init__(weight_decay)
        self.hyper_params = hyper_params
        self.input_size = input_size

    def __feat_ex(self, model, id):
        left_branch = keras.models.Sequential()
        conv_a = self.__conv2d('conv_{}a'.format(id))
        left_branch.add(conv_a)
        conv_b = self.__conv2d('conv_{}b'.format(id))
        left_branch.add(conv_b)

        right_branch = keras.models.Sequential()
        pool_a = self.__max_pooling2d('pool_{}a'.format(id))
        right_branch.add(pool_a)
        conv_c = self.__conv2d('conv_{}c'.format(id))
        right_branch.add(conv_c)

        concat = keras.layers.Merge([left_branch, right_branch], mode='concat')
        model.add(concat)

        pool_b = self.__max_pooling2d(concat, 'pool_{}b'.format(id))
        model.add(pool_b)

        return model

    def __conv2d(self, conv_name, first_layer=False):
        conv_hp = self.hyper_params[self.HP_CONV][conv_name]
        # mandatory
        kernel_size = conv_hp[self.HP_CONV_KERNEL_SIZE]
        n_filters = conv_hp[self.HP_CONV_N_FILTERS]
        # optional
        strides = conv_hp.get(self.HP_CONV_STRIDES, (1, 1))
        activation_fn = conv_hp.get(self.HP_CONV_ACTIVATION_FN, 'relu')
        regularizer = conv_hp.get(self.HP_CONV_REGL_FN, keras.regularizers.l2(self._weight_decay))
        padding_type = conv_hp.get(self.HP_CONV_PADDING_TYPE, 'same')

        # def: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d
        if first_layer:
            conv2d = keras.layers.Conv2D(
                input_shape=self.input_size,
                nb_filter=n_filters,
                nb_row=kernel_size[0],
                nb_col=kernel_size[1],
                init='glorot_uniform',
                activation=activation_fn,
                border_mode=padding_type,
                subsample=strides,
                W_regularizer=regularizer,
                dim_ordering='tf',
                bias=True
            )
        else:
            conv2d = keras.layers.Conv2D(
                nb_filter=n_filters,
                nb_row=kernel_size[0],
                nb_col=kernel_size[1],
                init='glorot_uniform',
                activation=activation_fn,
                border_mode=padding_type,
                subsample=strides,
                W_regularizer=regularizer,
                dim_ordering='tf',
                bias=True
            )
        return conv2d

    def __max_pooling2d(self, max_pool_name):
        max_pool_hp = self.hyper_params[self.HP_MP][max_pool_name]
        # mandatory
        pool_size = max_pool_hp[self.HP_MP_POOL_SIZE]
        # optional
        strides = max_pool_hp.get(self.HP_MP_STRIDES, pool_size)
        padding_type = max_pool_hp.get(self.HP_MP_PADDING_TYPE, 'same')

        reduced_weights = keras.layers.pooling.MaxPooling2D(
            # custom
            pool_size,
            strides,
            border_mode=padding_type,
            # defaults
            dim_ordering='tf'
        )
        return reduced_weights

    def __fully_connected(self, fc_name, n_outs=None):
        fc_hp = self.hyper_params[self.HP_FC][fc_name]
        n_outs = n_outs if n_outs else fc_hp[self.HP_FC_N_OUTPUTS]

        # since 'fc_1' is optional
        if not n_outs:
            return None, True

        # optional
        regularizer = fc_hp.get(self.HP_FC_REGL_FN, keras.regularizers.l2(self._weight_decay))
        activation_fn = fc_hp.get(self.HP_FC_ACTIVATION_FN, 'relu')

        dense_layer = keras.layers.Dense(
            # custom
            output_dim=n_outs,
            init='glorot_uniform',
            activation=activation_fn,
            W_regularizer=regularizer,
            bias=True,
        )
        return dense_layer, False

    def __batch_normalization(self):
        batch_norm = keras.layers.normalization.BatchNormalization(
            # default
            epsilon=0.001,
            mode=0,
            axis=-1,
            momentum=0.99,
            weights=None,
            beta_init='zero',
            gamma_init='one',
            gamma_regularizer=None,
            beta_regularizer=None
        )
        return batch_norm

    def __dropout(self, after_layer_name):
        after_layer_hp = self.hyper_params[self.HP_FC][after_layer_name]
        # mandatory
        dropout_rate = after_layer_hp[self.HP_FC_DROPOUT_RATE]

        out = keras.layers.core.Dropout(
            # custom
            p=dropout_rate,
        )
        return out

    def inference(self, inputs, labels, is_training=True):
        model = keras.models.Sequential()
        # level 1
        conv_1 = self.__conv2d('conv_1', first_layer=True)
        model.add(conv_1)
        pool_1 = self.__max_pooling2d('pool_1')
        model.add(pool_1)

        # we are using batch normalisation instead of lrn
        lrn_1 = self.__batch_normalization()
        model.add(lrn_1)

        # level 2
        model = self.__feat_ex(model, '2')

        # level 3
        model = self.__feat_ex(model, '3')

        # optional fully connected which is not in the paper
        fc_1, skip_fc_1 = self.__fully_connected('fc_1')
        model.add(fc_1)

        if not skip_fc_1:
            logging.warning("You have added an optional fully connected layer to the network which is not " +
                            "a part of the original DeXepression network. A dropout will also be applied" +
                            "after the fully connected layer.")
            fc_1 = self.__dropout('fc_1')
            model.add(fc_1)

        # classification layer
        classifier, _ = self.__fully_connected('out')
        model.add(classifier)

        return model

    # def loss(self, predictions, labels):
    #     # note tf.nn.softmax_cross_entropy_with_logits expects pred to be unscaled,
    #     # since it performs a softmax on logits internally for efficiency. Otherwise it is same as -
    #     #  -(y * log(softmax(pred)))
    #     # note the 'minus' at the front of the above equation
    #     # do not call this op with the output of softmax, as it will produce incorrect results.
    #     self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels))
    #     return self._loss
    #
    # def metrics(self, predictions, labels):
    #     # Evaluating the model
    #     true_positives = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    #     avg_accuracy = tf.reduce_mean(tf.cast(true_positives, tf.float32))
    #     return {'accuracy': avg_accuracy}
