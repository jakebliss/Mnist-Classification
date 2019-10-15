import keras
from mlxtend.data import loadlocal_mnist
from keras.models import Sequential 
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K

x_train, y_train = loadlocal_mnist(
  images_path='./train-images-idx3-ubyte',
  labels_path='./train-labels-idx1-ubyte')

x_test, y_test = loadlocal_mnist(
  images_path='./t10k-images-idx3-ubyte',
  labels_path='./t10k-labels-idx1-ubyte')

print('Dimensions: %s x %s' % (x_train.shape[0], x_train.shape[1]))
print('Dimensions: %s' % (y_train.shape[0]))
print('Dimensions: %s x %s' % (x_test.shape[0], x_test.shape[1]))
print('Dimensions: %s' % (y_test.shape[0]))
