import numpy as np
import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout
from sklearn.utils import shuffle

## IMPORT IMAGES AND LABELS ##################################################

train_images = mnist.train_images() 
test_images = mnist.test_images() 
train_labels = mnist.train_labels()
test_labels = mnist.test_labels()

train_images_rotate_90 = np.rot90(train_images,1,(1,2))
train_images_rotate_180 = np.rot90(train_images,2,(1,2))
train_images_rotate_270 = np.rot90(train_images,3,(1,2))
train_images_flip_x = np.flip(train_images,1)
train_images_flip_y = np.flip(train_images,2)

test_images_rotate_90 = np.rot90(test_images,1,(1,2))
test_images_rotate_180 = np.rot90(test_images,2,(1,2))
test_images_rotate_270 = np.rot90(test_images,3,(1,2))
test_images_flip_x = np.flip(test_images,1)
test_images_flip_y = np.flip(test_images,2)

train_images_full = np.concatenate((train_images, 
                                    train_images_rotate_90, 
                                    train_images_rotate_180, 
                                    train_images_rotate_270, 
                                    train_images_flip_x, 
                                    train_images_flip_y), 0)

test_images_full = np.concatenate((test_images, 
                                    test_images_rotate_90, 
                                    test_images_rotate_180, 
                                    test_images_rotate_270, 
                                    test_images_flip_x, 
                                    test_images_flip_y), 0)

train_labels_full = np.concatenate((train_labels, train_labels, train_labels,
                                    train_labels, train_labels, train_labels),0)

test_labels_full = np.concatenate((test_labels, test_labels, test_labels,
                                    test_labels, test_labels, test_labels),0)

train_images_shuffled, train_labels_shuffled = shuffle(np.array(train_images_full),
                                                       np.array(train_labels_full))

test_images_shuffled, test_labels_shuffled = shuffle(np.array(test_images_full),
                                                       np.array(test_labels_full))



## NORMALIZE IMAGES ##########################################################

train_images_shuffled = (train_images_shuffled / 255) - 0.5
test_images_shuffled = (test_images_shuffled / 255) - 0.5

## FLATTEN IMAGES ############################################################

train_images_shuffled = train_images_shuffled.reshape((-1, 784)) 
test_images_shuffled = test_images_shuffled.reshape((-1, 784))

#-1 to indicate we want a matrix of size i x 784, where i is to be found

print(train_images_shuffled.shape) # (360000, 784)
print(test_images_shuffled.shape)  # (60000, 784)

## BUILDING THE MODEL ########################################################

model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dropout(0.25),
  Dense(10, activation='softmax'),
])

## COMPILING THE MODEL #######################################################

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

## TRAINING THE MODEL ########################################################

history = model.fit(
  train_images_shuffled, # training data
  to_categorical(train_labels_shuffled), # training targets
  epochs=25,
  batch_size=32,
  validation_data=(test_images_shuffled, to_categorical(test_labels_shuffled)),
  validation_split=0.97222222222,
)

## TESTING THE MODEL #########################################################

test_loss, test_acc = model.evaluate(test_images_shuffled, to_categorical(test_labels_shuffled))
print('Test accuracy:', test_acc)
print('Test loss: ', test_loss)

## SAVING THE MODEL ##########################################################

model.save_weights('model.h5')

## USING THE MODEL ###########################################################

i = 10
# pick how many predictions to compare.

predictions = model.predict(test_images_shuffled[:i])
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]
print(test_labels_shuffled[:i]) # [7, 2, 1, 0, 4]

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

## ACTIVATION VARIATION ######################################################

simple_accuracy = np.array(history.history['accuracy'])
simple_val_accuracy = np.array(history.history['val_accuracy'])
simple_loss = np.array(history.history['loss'])
simple_val_loss = np.array(history.history['val_loss'])





















