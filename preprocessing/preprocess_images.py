import numpy as np
import scipy
import scipy.ndimage.interpolation
import matplotlib.pyplot as plt

def crop_image(X, loc, size):
  H, W, C = X.shape
  x, y = loc
  out = np.zeros((size, size, C))
  out = X[x-size/2:x+size/2, y-size/2:y+size/2, :]
  return out

def resize_image(X, size):
  H, W, C = X.shape
  out = np.zeros((size, size, C))
  scale = float(size) / float(H)
  out = scipy.ndimage.interpolation.zoom(X, (scale, scale, 1))
  return out

def get_glimpse(Xin, loc, size, scale, depth=2):
  H, W, C = Xin.shape
  x, y = loc
  pad_width = ( size * (scale ** depth) ) / 2
  X = np.pad(Xin, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), 'constant', constant_values = ((0, 0), (0, 0), (0, 0)))
  loc = (x + pad_width, y + pad_width)
  crop_size = size
  out = np.zeros((size, size, depth*C))
  for d in xrange(depth) :
    cropped_image = crop_image(X, loc, crop_size)
    plt.imshow(cropped_image)
    plt.show()
    resized_image = resize_image(cropped_image, size)
    plt.imshow(resized_image)
    plt.show()
    out[:,:,d*C:(d+1)*C] = resized_image
    crop_size = crop_size * scale
  return out
