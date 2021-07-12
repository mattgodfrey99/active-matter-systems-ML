# Active Matter Systems with Machine Learning

Using Machine Learning to model collective and adversarial behaviours in active matter systems. Completed as part of a summer internship at the University of Nottingham.

The two folders are split between MNIST and the Brownian motion aspects. MNIST was simply me getting used to machine learning using keras, and Brownian motion contains the code used for modelling an active matter system and applying machine learning to it. Descriptions for how the code works is described well within in file, or within the pdf write up of my summer internship. It is worth noting this write up was informal, and as such lacks a reference list and is not complete.

## For MNIST:

- MNIST_NN.py is a normal neural network trained on the MNIST dataset to detect the digits shown in images.

- MNIST_CNN.py is a convolutional neural network trained on the MNIST dataset to detect the digits shown in images.

- MNIST_NN_varied.py MNIST_CNN_varied.py is a normal neural network and convolutional neural network trained on the MNIST dataset to detect the digits shown in images respectively. The "varied" is due to me transforming the MNIST data to be more varied. For example, I supplied the networks with the MNIST data once it had been flipped, inverted, or rotated.

## For Brownian Motion:

- random_walk_simple.py is the code used for just normal brownian motion. It models particles randomly moving according to brownian motion. From it, flocking or clustering of all particles can be induced. This will later be used to collect training data for the neural network.

- brownian_data_collector.py is the above code, but rewritten to allow for data to be collected. This data can be the positions of all particles at each time step, their distances to each other, etc.

- brownian_NN.py is my most recent version of a neural network which has been trained on data supplied from the above code. 

- browian_integrated_sections.py is the most recent version of a code which is supplied with inital positions for all particles, and then a neural network model (from above code) is used to predict which positions all particles should take next. Depending on if the system was trained on flocking, brownian, or clustering data, the output of this code will vary.
