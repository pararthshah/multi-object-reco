# Python script to test basic convolution/max pooling operations using THeano
# Works on a minibatch of images
import numpy as np
import pylab
from PIL import Image
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

rng = np.random.RandomState(2345)

class(maxPoolLayer):
  def __init__(self, inputMinibatch, poolShape = (2, 2))
  
    self.inputMinibatch = inputMinibatch

    # Symbolic expression to perform the actual 2D convolution
    self.outMinibatch = downsample.max_pool_2d(inputMinibatch, poolShape, ignore_norder=True)
