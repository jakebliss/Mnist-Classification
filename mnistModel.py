import keras
from mlxtend.data import loadlocal_mnist
from keras.models import Sequential 
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 12 

ROWS, COLS = 28, 28

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
  x_train = x_train.reshape(x_train.shape[0], 1, ROWS, COLS)
  x_test = x_test.reshape(x_test.shape[0], 1, ROWS, COLS)
  input_shape = (1, ROWS, COLS)
else:
  x_train = x_train.reshape(x_train.shape[0], ROWS, COLS, 1)
  x_test = x_test.reshape(x_test.shape[0], ROWS, COLS, 1)
  input_shape = (ROWS, COLS, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Transform labels into binary encoding 
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
print(y_train)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test,y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])





