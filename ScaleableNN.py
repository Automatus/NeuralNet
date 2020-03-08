# NeuralNet, custom inputs, layers and outputs
# neurons are spiking neurons
# using numpy arrays

import numpy as np

#treshold or bias
b = 0.5

#learning rate
step = 0.1

#inputls and hidden neurons and outputsand weights
import numpy as np

inputval = int(input("number of inputs"))
hl = int(input("number of hidden layers"))
nl = int(input("number of neurons in each hidden layer"))
outputval = int(input("number of outputs"))

w = np.zeros((hl, nl, nl))
wi = np.zeros((inputval, nl))
wo = np.zeros((nl, outputval))
inputs = np.zeros((inputval))
neurons = np.zeros((hl, nl))
outputs = np.zeros((outputval))
youtputs = np.zeros((outputval))

while True:
    print("Please give boolean input")
    i = 0
    for item in inputs:
        inputs[i] = int(input("input for input " + str(i)))
        i =+ 1

    print("calculating answer")
    i = -1
    while i <= hl + 1:
        if i == -1:
            neurons[0,] = (inputs * wi) > b
        if i == hl + 1:
            outputs = neurons[hl-1,] * w[i,]
        else:
            neurons[i,] = (neurons[i-1,] * w[i,]) > b
        i =+1

    print("calculated output:", outputs)

    print("Please give desired answers:")
    i = 0
    for item in youtputs:
        youtputs[i] = int(input("desired input ", i))
        i =+1

    #learning with kind of backpropagation without derivatives comes here
