import numpy as np
import scipy 
from preprocess_images import *
import PIL.Image as img
import matplotlib.pyplot as plt

inp_image = img.open('/afs/.ir.stanford.edu/users/c/d/cduvedi/CS231N/assignment2/kitten.jpg')
inp_image.load()
data = np.asarray(inp_image)
#cropped_image = crop_image(data, (130, 200), 120)
#print cropped_image.shape
#
#resized_crop = resize_image(cropped_image, 60)
#print resized_crop.shape

#plt.imshow(data)
#plt.show()
#plt.imshow(cropped_image)
#plt.show()
#plt.imshow(resized_crop)
#plt.show()

glimpse = get_glimpse(data, (130, 120), 50, 2)
print glimpse.shape
