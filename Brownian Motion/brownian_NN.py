import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras
from keras.utils import to_categorical

## NORMALIZE IMAGES ##########################################################

# all images and labels imported, so obviously wont run without data. This is
# designed for running data with m iterations, n particles, 4 parameters
# (size of test data array is [m,n,4]). 

L = 5
# length of 'box' that houses particles
n = 10
# number of particles

train_images[:,:,0:2] = train_images[:,:,0:2]/(L/2)
# normalise [x,y] from -L:L to -1:1.
train_images[:,:,2:3] = train_images[:,:,2:3]/(np.pi)
# normalise theta value from -pi:pi to -1:1
train_images[:,:,3:4] = (train_images[:,:,3:4]/((L/2)*np.sqrt(2))*2)-1
# normalise distance value from 0:sqrt(2)(L/2) to -1:1

test_images[:,:,0:2] = test_images[:,:,0:2]/L
test_images[:,:,2:3] = test_images[:,:,2:3]/(np.pi)
test_images[:,:,3:4] = (test_images[:,:,3:4]/(L*np.sqrt(2))*2)-1

## FLATTEN IMAGES ############################################################

train_images = train_images.reshape((-1, 4*(n-1))) 
# reshape so each input is a single dimension
# 4*(n-1) due to 4 parameters, adn n-1 particles (since one is redundant info)
test_images = test_images.reshape((-1, 4*(n-1)))

## BUILDING THE MODEL ########################################################

#initializer = keras.initializers.RandomNormal(mean=0., stddev=0.1)

model = Sequential([
  Dense(500, activation='tanh', input_shape=(4*(n-1),)),
  Dense(64, activation='tanh'),
  #Dense(8, activation='tanh'),
  Dropout(0.25),
  Dense(12, activation='softmax'),
])

# only all tanh as I was just trying different variations

## COMPILING THE MODEL #######################################################

opt = keras.optimizers.Adam(learning_rate=0.0001)
# seeing how learning rate effected it

model.compile(
  optimizer=opt,
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

## TRAINING THE MODEL ########################################################

history = model.fit(
  train_images, # training data
  thing, # training targets
  epochs=50,
  batch_size=256,
  #validation_data=(test_images, test_labels),
  shuffle=True,
  validation_split=0.2,
)

## SAVING THE MODEL ##########################################################

# save the model to disk.
model.save_weights('model.h5')

# load the model from disk later using:
# model.load_weights('model.h5')

## USING THE MODEL ###########################################################

i =100
# which 'image' to see prediction for

predictions = model.predict(train_images[i:i+1,:])
# single prediction
print('Prediction: ',predictions) 
print(np.argmax(predictions, axis=1))
actual = train_labels[i]
# actual label
print('Actual: ', actual)

#print('Absolute difference: ', abs(test_labels[i]) - abs(predictions))
# difference between prediction and label

plt.figure(0)
plt.plot(train_labels[:i])
plt.plot(model.predict(train_images[:i]))
plt.title('Predicted vs. Actual')
plt.ylabel('delta_theta')
plt.xlabel('Example no.')
plt.legend(['Actual', 'Predicted'], loc='best')
# plot to compare real and predicted values 

## PLOTS #####################################################################

# mean squared error vs epoch number

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Error')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['Trained', 'Test'], loc='best')
plt.show()

## ACTIVATION VARIATION ######################################################

simple_mse = np.array(history.history['mean_squared_error'])
simple_val_mse = np.array(history.history['val_mean_squared_error'])

