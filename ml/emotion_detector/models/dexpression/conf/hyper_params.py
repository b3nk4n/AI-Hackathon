# -*- coding: utf-8 -*-

from __future__ import absolute_import

from models import DexpressionNet as DN

# example
# here goes some comments if needed
# dexpression_hp_ddmm_hhmm = {
# ...
# }

# DO NOT FORGET TO SET IT TO dex_hyper_params at the end of this file!!!
dexpression_hp_2406_1100 = {
    DN.HP_CONV: {
        'conv_1': {
            DN.HP_CONV_KERNEL_SIZE: (7, 7),
            DN.HP_CONV_STRIDES: (2, 2),
            DN.HP_CONV_N_FILTERS: 64,
        },
        'conv_2a': {
            DN.HP_CONV_KERNEL_SIZE: (1, 1),
            DN.HP_CONV_STRIDES: (1, 1),
            DN.HP_CONV_N_FILTERS: 96,
        },
        'conv_2b': {
            DN.HP_CONV_KERNEL_SIZE: (3, 3),
            DN.HP_CONV_STRIDES: (1, 1),
            DN.HP_CONV_N_FILTERS: 208,
        },
        'conv_2c': {
            DN.HP_CONV_KERNEL_SIZE: (1, 1),
            DN.HP_CONV_STRIDES: (1, 1),
            DN.HP_CONV_N_FILTERS: 64,
        },
        # should be same as conv_2a
        'conv_3a': {
            DN.HP_CONV_KERNEL_SIZE: (1, 1),
            DN.HP_CONV_STRIDES: (1, 1),
            DN.HP_CONV_N_FILTERS: 96,
        },
        # should be same as conv_2b
        'conv_3b': {
            DN.HP_CONV_KERNEL_SIZE: (3, 3),
            DN.HP_CONV_STRIDES: (1, 1),
            DN.HP_CONV_N_FILTERS: 208,
        },
        # should be same as conv_2c
        'conv_3c': {
            DN.HP_CONV_KERNEL_SIZE: (1, 1),
            DN.HP_CONV_STRIDES: (1, 1),
            DN.HP_CONV_N_FILTERS: 64,
        }
    },

    DN.HP_MP: {
        'pool_1': {
            DN.HP_MP_POOL_SIZE: (3, 3),
            DN.HP_MP_STRIDES: (2, 2),
        },
        'pool_2a': {
            DN.HP_MP_POOL_SIZE: (3, 3),
            DN.HP_MP_STRIDES: (1, 1),
        },
        'pool_2b': {
            DN.HP_MP_POOL_SIZE: (3, 3),
            DN.HP_MP_STRIDES: (2, 2),
        },
        # should be same as pool_2a
        'pool_3a': {
            DN.HP_MP_POOL_SIZE: (3, 3),
            DN.HP_MP_STRIDES: (1, 1),
        },
        # should be same as pool_2b
        'pool_3b': {
            DN.HP_MP_POOL_SIZE: (3, 3),
            DN.HP_MP_STRIDES: (2, 2),
        },
    },

    DN.HP_FC: {
        # this one is not in the original dexepression
        'fc_1': {
            DN.HP_FC_N_OUTPUTS: 1024,
            DN.HP_FC_DROPOUT_RATE: 0.5
        },
        'out': {
            # the n_outputs for out layer is feed separately from within the classifier
        }
    }
}


dex_hyper_params = dexpression_hp_2406_1100
