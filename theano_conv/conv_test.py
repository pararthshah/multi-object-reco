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

# Symbolic variable for the input minibacth of images
# Number of images * Number of channels * Height * Width
inputMinibatch = T.tensor4('inputMinibatch')

# Initialize a set of convolution filter weights
# Number of filters * Image depth * filterHeight * filterWidth
wShape = (2, 3, 5, 5)
wBound = np.sqrt(3 * 5 * 5)
W = theano.shared(np.asarray(
			rng.uniform(
				low = -1 / wBound,
				high = 1 / wBound,
				size = wShape),
			),
			#d_type=inputMinibatch.dtype), 
			name='W')

bShape = (2, )
b = theano.shared(np.asarray(
			np.zeros(bShape),
			),
			#d_type = inputMinibatch.dtype), 
		name = 'b')

# Perform the actual 2D convolution
convImg = conv.conv2d(inputMinibatch, W)
activation = lambda x: x * (x > 0)
# Add the bias and apply the activation function
outMinibatch = activation(convImg + b.dimshuffle('x', 0, 'x', 'x'))

# Create a theano function
convFunction = theano.function([inputMinibatch], outMinibatch)

img = Image.open(open('/afs/.ir/users/c/d/cduvedi/Theano/doc/images/3wolfmoon.jpg'))
img = np.asarray(img, dtype=theano.config.floatX) / 256.

# The image is now read as a 639 * 516 * 3 array
# Reshape it to 1 * 3 * 639 * 516
imgTensor = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, 639, 516)

filteredImage = convFunction(imgTensor)

# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filteredImage[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filteredImage[0, 1, :, :])
pylab.show()
