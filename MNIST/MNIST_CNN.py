import numpy as np
import matplotlib.pyplot as plt
import mnist
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.layers import Dropout

## IMPORT IMAGES AND LABELS ##################################################

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

## NORMALIZE IMAGES ##########################################################

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

## RESHAPE IMAGES ############################################################

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)
# reshape each image from (28, 28) to (28, 28, 1)

## BUILDING THE MODEL ########################################################

num_filters = 8
filter_size = 3
pool_size = 2
# these are from the other CNN we did manually

model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28,28,1),activation='relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
    Dense(10, activation='softmax'),
    ])

## COMPILING THE MODEL #######################################################

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

## TRAINING THE MODEL ########################################################

history = model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=30,
  validation_data=(test_images, to_categorical(test_labels)),
)

## SAVING THE MODEL ##########################################################

model.save_weights('model.h5')

## TESTING THE MODEL #########################################################

predictions = model.predict(test_images[:5])
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]
print(test_labels[:5]) # [7, 2, 1, 0, 4]

## PLOTS #####################################################################

# accuracy vs epoch number

plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Trained', 'Test'], loc='best')
plt.grid()
plt.show()

# loss vs epoch number

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Trained', 'Test'], loc='best')
plt.grid()
plt.show()

## SAVING LOGS ###############################################################

CNN_accuracy = np.array(history.history['accuracy'])
CNN_val_accuracy = np.array(history.history['val_accuracy'])
CNN_loss = np.array(history.history['loss'])
CNN_val_loss = np.array(history.history['val_loss'])


























