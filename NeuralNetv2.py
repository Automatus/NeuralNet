# NeuralNet, custom inputs, layers and outputs
# neurons are spiking neurons
# learning algorithm compares desired output with actual output
# using numpy arrays

import numpy as np

b = 0.5  # treshold/bias
step = 0.1  # learning rate
# inputs and hidden neurons and outputs and weights:
inputval = int(input("number of inputs"))
hl = int(input("number of hidden layers"))
nl = int(input("number of neurons in each hidden layer"))
outputval = int(input("number of outputs"))
w = np.zeros((hl, nl, nl))
wi = np.zeros((inputval, nl))
wo = np.zeros((nl, outputval))
inputs = np.zeros(inputval)
neurons = np.zeros((hl, nl))
outputs = np.zeros(outputval)
youtputs = np.zeros(outputval)
spll = np.zeros(nl)  # spiking neurons in last layer that have a connection to the output that is being updated

while True:
    print("NeuralNet v2 by Automatus")

    # Giving the Input
    print("Please give input")
    i = 0
    for item in inputs:
        inputs[i] = int(input("input for input " + str(i)))
        i = +1
    print("Inputs:", inputs)

    # Calculating the output
    print("calculating answer")
    i = -1  # = current layer
    while i <= hl + 1:
        if i == -1:
            neurons[0, :] = (inputs * wi) > b
        if i == hl + 1:
            outputs = neurons[hl-1, :] * wo
        else:
            neurons[i, :] = (neurons[i-1, :] * w[i, :]) > b
        i = +1
    print("calculated answer:", outputs)

    # Getting y/desired output
    print("Please give desired answers:")
    i = 0
    for item in youtputs:
        youtputs[i] = input("desired output for input" + str(i))
        i = +1

    # learning with backpropagation without derivatives
    print("Updating weigths")
    i = 0  # index for output
    for item in outputs:  # iterating through outputs
        if 1 not in neurons[-1, :] and item:  # if no active neurons in last layer
            s = -2  # index for layer
            for layer in neurons:  # iterating through layers
                stop = False
                if not stop:
                    if 1 in neurons[s, :]:  # if active neuron present in layer s
                        stop = True  # stop iterating through layers
                        t = 0  # index for neuron
                        for neuron in neurons[s, :]:  # iterating through neurons
                            if neuron:  # if neuron is active
                                w[s, :, t] = w[s, :, t] + step  # strenghten weights/connections
                            t = +1
                s = -1
        else:  # if active neurons in last layer are present
            f_error = youtputs[i] - outputs[i]  # calculate error
            j = -1  # index for layer of neurons
            k = 0  # index for neuron in layer j
            for neuron in neurons[j, :]:
                if neuron:  # if neuron is firing/1/True
                    wo[i, k] = wo[i, k] + f_error * step
                    spll[k] = True
                k = +1
            j = -1
            while j > -hl:  # while we are iterating neural layers
                k = 0
                for neuron in neurons[j, :]:
                    if neuron:
                        z = 0
                        for thing in spll:
                            if spll[z]:
                                w[j, z, k] = w[j, z, k] + f_error * step
                    k = +1
            x = 0
            for innn in inputs:
                if innn:
                    q = 0
                    for it in spll:
                        if spll[q]:
                            wi[q, x] = wi[q, x] + f_error * step
                        q = +1
                x = +1
        i = +1

    print("Weights updated,  new weights:")
    print(wi)
    print(w)
    print(wo)
