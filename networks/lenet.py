from layers import conv_layer
from layers import max_pool_layer
from layers import fc_layer
from networks import lenet_config
from networks.lenet_config import *
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
rng = np.random.RandomState(2345)

def eval_lenet(batchSize, learningRate = 0.01):
  # Symbolic variables for training examples and labels
  X = T.matrix('X')
  y = T.matrix('y')

  # Build the model using the other classes
  layer0Shape = (batchSize, imageSize[2], imageSize[1], imageSize[0])
  layer0.input = X.reshape(layer0Shape)

  layer0 = convLayer(rng, X, filter0Shape, layer0Shape)

  layer1Input = layer0.outMinibatch

  layer1 = maxPoolLayer(layer1Input, poolSize)

  layer2Input = layer1.outMinibatch
  layer2Shape = (batchSize, filter0Shape[0], imageSize[1], imageSize[0])
  layer2 = convLayer(rng, layer2Input, filter2Shape, layer2Shape)

  layer3Input = layer2.outMinibatch

  layer3 = maxPoolLayer(layer1Input, poolSize)

  layer4Input = layer3.outMinibatch.reshape(layer3.outMinibatch.shape[0], -1)

  layer4 = fc_layer(rng, layer4Input, layer4Input.shape[0], numFCneurons)

  layer5Input = layer4.output.reshape(layer4.outMinibatch.shape[0], -1)

  layer5 = fc_layer(rng, layer5Input, layer5Input.shape[0], numClasses, activation = 'softmax', y)

  cost = layer5.loss
  yPred = layer5.y_pred
  errors = T.mean(T.neq(yPred, y))

  # Aggregate the parameters from all the layers
  params = layer0.params + layer1.params + layer2.params + layer3.params +layer3.params + layer4.params + layer5.params

  # Find the gradients of all the parameters
  grads = T.grad(cost, params)

  updates = [
      (param_i, param_i - learningRate * grad_i )
      for param_i, grad_i in zip(params, grads)
      ]

  train_CNN = theano.function(
      [x, y],
      cost,
      updates = updates
      )

  validate_CNN = theano.function(
      [x, y],
      errors,
      )

  test_CNN = theano.function(
      [x, y],
      errors,
      )
if __name__ == '__main__':
  eval_lenet(20)
