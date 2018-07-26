
"""
@author: Douglas Amante
"""

#Network Feed Forward - MLP

#Supervised learning

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

#Passing the dimensions of the input and target vectors

dataset = SupervisedDataSet(2,1)


#Number of entries =2, Number of output = 1 

dataset.addSample([1,1],[0])

dataset.addSample([1,0],[1])

dataset.addSample([0,1],[1])

dataset.addSample([0,0],[0])

#Second parameter is the number of neurons in the middle layer
#Bias, unit value, for extra input

# input layer, layer 1, output layer,
#example (2,4,7,1) = input = 2 neurons, hidden layer 1 = 4 neurons, hidden layer 2 = 7 layers, output = 1 neuron

network = buildNetwork(dataset.indim, 4, dataset.outdim, bias=True)

# learning rate = learning rate = speed controlling weight weights (ETA), minimum local, overall maximum
#momentum = Rate of instability, increases convergence

trainer = BackpropTrainer(network, dataset, learningrate=0.01, momentum=0.99)

#backproperation algorithm = Backprop Trainer, using the sigmoid function

#for epoca in range (1000): #treine per 1000 epochs
  # trainer.train ()

#train to convergence
#trainer.trainUntilConvergence


# number of time, to test the network cycles, 1000 training cycles

trainer.trainEpochs(1000)

test_data = SupervisedDataSet(2,1)

test_data.addSample([1,1],[0])

test_data.addSample([1,0],[1])

test_data.addSample([0,1],[1])

test_data.addSample([0,0],[0])

trainer.testOnData(test_data, verbose=True)
