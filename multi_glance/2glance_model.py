from deep_rl.layers import *
import numpy as np
import theano
from theano import theano.tensor as T
import matplotlib.pyplot as plt

class glance_network(Object):
  def __init__(self, width, height, learning_rate, decay, momentum=0, batch_size=32, num_glimpses=16):
    self._batch_size = batch_size
    self._img_width = width
    self._img_height = height
    self.learning_rate = learning_rate
    self.decay = decay
    self.momentum = momentum
    self.scale_input_by = 255.0
    self.num_glances = num_glances

    # Layers for the coarse glance
    self.layers = []
    self.layers.append(layers.Input2DLayer(self._batch_size,
                                               self._img_height,
                                               self._img_width,
                                               self.scale_input_by))

    self.layers.append(layers.StridedConv2DLayer(self.layers[-1],
                                                         n_filters=16,
                                                         filter_width=8,
                                                         filter_height=8,
                                                         stride_x=4,
                                                         stride_y=4,
                                                         weights_std=.01,
                                                         init_bias_value=0.01))

    self.layers.append(layers.StridedConv2DLayer(self.layers[-1],
                                                         n_filters=32,
                                                         filter_width=4,
                                                         filter_height=4,
                                                         stride_x=2,
                                                         stride_y=2,
                                                         weights_std=.01,
                                                         init_bias_value=0.01))

    self.layers.append(layers.DenseLayer(self.layers[-1],
                                                   n_outputs=256,
                                                   weights_std=0.01,
                                                   init_bias_value=0.1,
                                                   dropout=0,
                                                   nonlinearity=layers.rectify))

    self.layers.append(layers.DenseLayer(self.layers[-1],
                                                   n_outputs=self.num_glances,
                                                   weights_std=0.01,
                                                   init_bias_value=0.1,
                                                   dropout=0,
                                                   nonlinearity=layers.rectify))

    self.a = T.nnet.softmax(T.dot(self.layers[-1].output()))
    self.y_pred = T.argmax(self.a, axis=1)

