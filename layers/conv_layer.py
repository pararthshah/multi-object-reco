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

class(convLayer):
  def __init__(self, rng, inputMinibatch, filterShape, imageShape)
  
  # Check that the number of channels implied in the image and the filters is the same
  assert imageShape[1] == filterShape[1]  
  self.inputMinibatch = inputMinibatch

  # Initialize a set of convolution filter weights
  # Number of filters * Image depth * filterHeight * filterWidth
  wBound = np.sqrt(filterShape[1] * filterShape[2] * filterShape[3])
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
                        np.zeros(filterShape[0]),
                  ),
                name = 'b')

  # Symbolic expression to perform the actual 2D convolution
  convImg = conv.conv2d(input = inpuMinibatch, filters=self.W, fitler_shape = filterShape, image_shape = imageShape)

  activation = lambda x: x * (x > 0)
  # Add the bias and apply the activation function
  self.outMinibatch = activation(convImg + b.dimshuffle('x', 0, 'x', 'x'))

  # Store the parameters for backprop
  self.params = [self.W, self.b]
