
# Copyright (c) 2015, Pararth Shah
# All rights reserved.

import numpy as np
import theano.tensor as T
import theano
import sys
import os
import cPickle as pickle

class DRAM(object):
    def __init__(self, num_actions, phi_length, width, height,
                 discount, learning_rate, decay, momentum=0,
                 batch_size=32):
        self._batch_size = batch_size
        self._num_input_features = phi_length
        self._phi_length = phi_length
        self._img_width = width
        self._img_height = height
        self._discount = discount
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.scale_input_by = 255.0
        
        self._model = \
            {'glimpse': self.init_glimpse_net(),
            'recurrent': self.init_recurrent_net(),
            'emission': self.init_emission_net(),
            'context': self.init_context_net(),
            'classification': self.init_classification_net()
            }

    def init_glimpse_net(self):
        return []

    def init_recurrent_net(self):
        return []

    def init_emission_net(self):
        return []

    def init_context_net(self):
        return []

    def init_classification_net(self):
        return []

    def train(self):
        return []