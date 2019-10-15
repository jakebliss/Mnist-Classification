import keras
from mlxtend.data import loadlocal_mnist
from keras.models import Sequential 
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 12 

IMG_ROWS, IMG_COLS = 28, 28

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

if K.image_data_format() == 'channels_first': 
  x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
  x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
  input_shape = (1, IMG_ROWS, IMG_COLS)
else:
  x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
  x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
  input_shape = (IMG_ROWS, IMG_COLS, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')