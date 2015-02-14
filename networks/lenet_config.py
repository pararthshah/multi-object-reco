from layers import conv_layer
from layers import max_pool_layer
from layers import fc_layer
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

# Gray scale 28*28 images for MNIST
imageSize = (28, 28, 1)

# Number of filters in each layer
numFilters = [20, 50]
# First layer filters
filter0Shape = (numFilters[0], imageSize[2], 3, 3)
poolSize = (2, 2)

filter2Shape = (numFilters[1], numFilters[0], 3, 3)

numFCneurons = 500
numClasses = 10
