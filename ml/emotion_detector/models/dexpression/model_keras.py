# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging
import coloredlogs

import keras
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense

from models.base import AbstractModel

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


# def create_model(image_size):
#     model = Sequential()
#     model.add(Conv2D(32, 3, 3, input_shape=(image_size, image_size, 1)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(32, 3, 3))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(64, 3, 3))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#     model.add(Dense(64))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(7))
#     model.add(Activation('sigmoid'))
#     return model


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

    def __init__(self, weight_decay, input_shape, hyper_params, n_classes):
        super(DexpressionNet, self).__init__(weight_decay)
        self.hyper_params = hyper_params
        self.input_shape = input_shape
        self.n_classes = n_classes

    def __feat_ex(self, id, last_layer):
        conv_a = self.__conv2d('conv_{}a'.format(id))(last_layer)
        conv_b = self.__conv2d('conv_{}b'.format(id))(conv_a)

        pool_a = self.__max_pooling2d('pool_{}a'.format(id))(last_layer)
        conv_c = self.__conv2d('conv_{}c'.format(id))(pool_a)

        concat = keras.layers.merge([conv_b, conv_c], mode='concat', concat_axis=-1)

        pool_b = self.__max_pooling2d('pool_{}b'.format(id))(concat)

        return pool_b

    def __conv2d(self, conv_name):
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
            bias=True,
            name=conv_name
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
            dim_ordering='tf',
            name=max_pool_name
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
            name=fc_name
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
            name='dropout_{}'.format(after_layer_name)
        )
        return out

    def inference(self, inputs, labels, is_training=True):
        inputs = keras.layers.Input(shape=self.input_shape)
        # level 1
        conv_1 = self.__conv2d('conv_1')(inputs)
        pool_1 = self.__max_pooling2d('pool_1')(conv_1)

        # we are using batch normalisation instead of lrn
        lrn_1 = self.__batch_normalization()(pool_1)

        # level 2
        feat_ex_2 = self.__feat_ex('2', lrn_1)

        # level 3
        feat_ex_3 = self.__feat_ex('3', feat_ex_2)
        flattened_feat_ex_3 = keras.layers.core.Flatten()(feat_ex_3)

        # optional fully connected which is not in the paper
        fc_1, skip_fc_1 = self.__fully_connected('fc_1')

        if not skip_fc_1:
            logging.warning("You have added an optional fully connected layer to the network which is not " +
                            "a part of the original DeXepression network. A dropout will also be applied" +
                            "after the fully connected layer.")
            fc_1 = fc_1(flattened_feat_ex_3)
            fc_1_w_dropout = self.__dropout('fc_1')(fc_1)

        # classification layer
        classifier, _ = self.__fully_connected('out', n_outs=self.n_classes)
        classifier = classifier(fc_1_w_dropout)

        # scale to probability distribution
        predictions = keras.layers.Activation('softmax')(classifier)

        # put the graph in a keras model
        model = keras.models.Model(input=inputs, output=predictions)

        return model

    def loss(self, predictions, labels):
        return None

    def metrics(self, predictions, labels):
        return None
