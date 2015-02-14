# Python script to test basic convolution/max pooling operations using THeano
# Works on a minibatch of images
import numpy as np
import pylab
from PIL import Image
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
rng = np.random.RandomState(2345)

class fcLayer(object):
  def __init__(self, rng, inputMinibatch, numIn, numOut, activation=None, y):
    # Softmax activation for the last layer
    self.softmax = T.nnet.softmax 

    self.inputMinibatch = inputMinibatch

    # Initialize a set of convolution filter weights
    # Number of filters * Image depth * filterHeight * filterWidth
    wBound = numIn + numOut
    self.W = theano.shared(np.asarray(
                          rng.uniform(
                                  low = -1 / wBound,
                                  high = 1 / wBound,
                                  size = filterShape),
                          ),
                          name='W')
    # Initialize the biases to zero
    bShape = (2, )
    b = theano.shared(np.asarray(
                          np.zeros(numOut),
                    ),
                  name = 'b')

    # Symbolic expression to perform the affine transform
    affineOut = T.dot(inputMinibatch, self.W) + self.b

    # Add the bias and apply the activation function
    if activation is 'softmax':
      self.loss = self.softax(affineOut, y)
    else:
      self.loss = affineOut
    self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

    # Store the parameters for backprop
    self.params = [self.W, self.b]
  def __softmax__(scores, y):
    self.p_y_given_x = T.nnet.softmax(scores)
    return -T.mean(T.log(sself.p_y_given_x)[T.arrange(y.shape[0]), y])

