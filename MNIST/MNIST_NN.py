import numpy as np
import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Dropout

## IMPORT IMAGES AND LABELS ##################################################

train_images = mnist.train_images() # handwritten numbers for learning
train_labels = mnist.train_labels() # what numbers each image represents

print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape) # (60000,)

# can do print(train_labels[i]) to see what i'th image in array actually is.
# can do print(train_images[i,:,:]) to see i'th image array.

test_images = mnist.test_images() 
#handwritten numbers for testing

test_labels = mnist.test_labels() 
#what numbers each image represents

print(test_images.shape) # (10000, 28, 28)
print(test_labels.shape) # (10000,)

# show handwritten number image: 
# plt.imshow(train_images[i]) 
# plt.show()

## NORMALIZE IMAGES ##########################################################

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# usually between 0 and 255, now between -0.5 and 0.5.

## FLATTEN IMAGES ############################################################

train_images = train_images.reshape((-1, 784)) 
test_images = test_images.reshape((-1, 784))

#-1 to indicate we want a matrix of size i x 784, where i is to be found

print(train_images.shape) # (60000, 784)
print(test_images.shape)  # (10000, 784)

## BUILDING THE MODEL ########################################################

model = Sequential([
  Dense(64, activation='relu', input_shape=(784,),kernel_initializer='Constant'),
  Dense(64, activation='relu'),
  Dropout(0.25),
  Dense(10, activation='softmax'),
])

# sequential: creates linear stack of layers.

# dense: fully connected layer (each neuron connected to every other neuron in
# next network).

# ReLU: x = 0 for x < 0, x = x for x > 0.

# softmax: turns arbitrary numbers into probabilities. Mathematical expression
# using exponentials bookmarked.

# first: flattened input layer (passing individual data to individual neuron,
# as opposed to sending a list (whole image array) to a neuron) with 64 
# neurons, 784 tells the network the shape of the input.

# second: dense layer with 64 neurons, with activation function ReLU.

# third: another dense layer with 10 neurons (number values between 0 and 9),
# and has activation function softmax (essentially probability the network
# thinks the image is a specific number).

## COMPILING THE MODEL #######################################################

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)


# optimizer: algorithm used to update the weights of the network. Adam is a 
# standard optimizer to use, there are many other though. Examples bookmarked.

# loss: this function is used to compute the amount that the model should try 
# to minimize during training. Cross-entropy loss increases as the predicted 
# probability diverges from the actual label. A perfect model would have a log
# loss of 0.

# metrics: we are interested in the accuracy; basically how low we can get the
# loss function to be.

## TRAINING THE MODEL ########################################################

history = model.fit(
  train_images, # training data
  to_categorical(train_labels), # training targets
  epochs=10,
  batch_size=32,
  validation_data=(test_images, to_categorical(test_labels)),
  #validation_split=0.66,
)

# epochs: how many times the model sees each image and corresponding label.
# It wont show the same types of images one after the other, but instead in 
# some other order. Basically the iterations over the entire dataset.
# Increasing the number of epochs does not necessarily increase the accuracy,
# the optimal number of epochs depends on your neural network. Increasing to
# 10 for this example increases the accuracy, but with diminishing returns.

# to_categorical: keras expects the training targets to be 10-dimensional
# vectors, ours are 1-dimensional (1,2,3...etc). This basically turns our 1-D 
# target into a 10-D target (e.g. 3 -> [0,0,0,1,0,0,0,0,0,0])

# batch_size: number of training examples utilized in one iteration (number of
# samples per gradient update).

# accuracy we get should be ~96.6%. This is what our current model thinks the 
# accuracy will be from what it has learned.

# from the tensorflow github page, the error 'WARNING:tensorflow:AutoGraph 
# could not transform...' seems to be normal and does not effect performance

## TESTING THE MODEL #########################################################

test_loss, test_acc = model.evaluate(test_images, to_categorical(test_labels))
print('Test accuracy:', test_acc)
print('Test loss: ', test_loss)

# evaluate(): returns an array containing the test loss followed by any
# metrics we specified (accuracy in this case).



## SAVING THE MODEL ##########################################################

# save the model to disk.
model.save_weights('model.h5')

# load the model from disk later using:
# model.load_weights('model.h5')

## USING THE MODEL ###########################################################

i = 5
# pick how many predictions to compare.

predictions = model.predict(test_images[:i])
#predicts on the first 5 test images. On its own, the array 'predictions' 
#will have 10 probabilities for each number. We refine that below.

print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]
# prints our model's predictions. np.argmax simply gets the highest value of
# the array, corresponding to most highest probability.

print(test_labels[:i]) # [7, 2, 1, 0, 4]
# checks our predictions against the actual value.
 
## PLOTS #####################################################################

# accuracy vs epoch number

plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Trained', 'Test'], loc='best')
plt.show()

# loss vs epoch number

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Trained', 'Test'], loc='best')
plt.show()

## ACTIVATION VARIATION ######################################################

simple_accuracy = np.array(history.history['accuracy'])
simple_val_accuracy = np.array(history.history['val_accuracy'])
simple_loss = np.array(history.history['loss'])
simple_val_loss = np.array(history.history['val_loss'])





















