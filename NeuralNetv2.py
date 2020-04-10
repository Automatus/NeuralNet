# NeuralNet v.2
# custom inputs, layers and outputs
# neurons are spiking neurons
# learning algorithm compares desired output with actual output
# using numpy arrays
# learning step depends on number of neurons and is randomized to avoid perfect parallelisms

import numpy as np
import random

print("NeuralNet v2 by Automatus")

b = 0.5  # treshold/bias
step = 0.3  # learning rate
# inputs and hidden neurons and outputs and weights:
number_of_inputs = int(input("number of inputs"))
number_of_hidden_layers = int(input("number of hidden layers"))
number_of_neurons_in_layer = int(input("number of neurons in each hidden layer"))
number_of_outputs = int(input("number of outputs"))
w = np.zeros((number_of_hidden_layers-1, number_of_neurons_in_layer, number_of_neurons_in_layer))  # weights between neurons
wi = np.zeros((number_of_inputs, number_of_neurons_in_layer))  # weights between inputs and first layer
wo = np.zeros((number_of_neurons_in_layer, number_of_outputs))  # weights between last layer and outputs
inputs = np.zeros((1, number_of_inputs))
neurons = np.zeros((number_of_hidden_layers, number_of_neurons_in_layer))
outputs = np.zeros((1, number_of_outputs))
youtputs = np.zeros((1, number_of_outputs))  # desired outputs
spll = np.zeros((1, number_of_neurons_in_layer))  # SPiking neurons in Last investigated Layer that have a connection to the output that is being updated

while True:
    # Giving the Input
    print("Please give input")
    i = 0
    for item in inputs[0, :]:
        inputs[0, i] = int(input("input for input " + str(i)))
        i += 1
    print("Inputs:", inputs)

    # Calculating the output
    print("calculating answer")
    i = 0  # = current layer
    while i <= number_of_hidden_layers:  # iterating through layers
        if i == 0:  # if calculating firts layer
            neurons[i, :] = (inputs.dot(wi)) > b  # dot product of inputs with weights for inputs and first layer, checks if it is bigger then treshold, then sets neuron as firing if true
        elif i == number_of_hidden_layers:  # if calculating last layer
            outputs = neurons[number_of_hidden_layers-1, :].reshape(1, number_of_neurons_in_layer).dot(wo)  # dot product of last neural layer with weigts for output
        else:  # if not calculating last or first layer, but layers in between
            neurons[i, :] = (neurons[i-1, :].dot(w[i-1, :])) > b  # dot product of current neural layer with weigths, checks if it is bigger then treshold, then sets neuron as firing if true
        i += 1
    print("calculated answer:", outputs)

    # Getting y/desired output
    print("Please give desired answers:")
    i = 0
    for item in youtputs[0, :]:
        youtputs[0, i] = input("desired output for input" + str(i))
        i += 1

    # learning with backpropagation (without derivatives)
    print("Updating weigths")
    i = 0  # index for output
    for item in youtputs[0, :]:  # iterating through outputs
        if 1 not in neurons[-1, :] and item:  # if no active neurons in last layer and desired output is 1
            s = -2  # index for layer
            stop = False
            while s >= -number_of_hidden_layers-1:  # iterating through layers
                if not stop:
                    if s == -number_of_hidden_layers-1 and 1 in inputs:  # if all neurons are negative, but positive input:
                        t = 0  # index for input
                        for bla in inputs[0, :]:  # iterating through inputs
                            if bla:  # if input is active
                                wi[t, :] = wi[t, :] + step*random.randint(80, 120)/100  # strenghten weights/connections to neurons
                            t += 1
                    elif 1 in neurons[s, :]:  # if active neuron present in layer s
                        stop = True  # stop iterating through layers
                        t = 0  # index for neuron
                        for neuron in neurons[s, :]:  # iterating through neurons
                            if neuron:  # if neuron is active
                                w[s+1, :, t] = w[s+1, :, t] + step/number_of_neurons_in_layer*random.randint(80, 120)/100  # strenghten weights/connections in proportion to number of neurons
                            t += 1
                s += -1
        else:  # if active neurons in last layer are present
            f_error = youtputs[0, i] - outputs[0, i]  # calculate error
            j = -1  # index for layer of neurons
            k = 0  # index for neuron in layer j
            for neuron in neurons[j, :]:
                if neuron:  # if neuron is firing/1/True
                    wo[k, i] = wo[k, i] + f_error * step/number_of_neurons_in_layer*random.randint(80, 120)/100  # change weight strenght in proportion to error and number of neurons
                    spll[0, k] = True  # remember spiking neuron for next iterations
                k += 1
            j = j - 1
            while j > -number_of_hidden_layers:  # while iterating through neural layers
                k = 0
                for neuron in neurons[j, :]:  # iterating through neurons in layer j
                    if neuron:  # if neuron fired
                        z = 0
                        for thing in spll[0, :]:  # iterating through remembered spiking neurons
                            if spll[0, z]:  # if neuron spiked
                                w[j+1, z, k] = w[j+1, z, k] + f_error * step/number_of_neurons_in_layer*random.randint(80, 120)/100  # change weight strenght in proportion to error and number of neurons
                            z += 1
                    k += 1
                j += -1
            x = 0
            for innn in inputs[0, :]:  # iterating through inputs
                if innn:  # if input is 1
                    q = 0
                    for it in spll[0, :]:  # iterating through remembered spiking neurons
                        if spll[0, q]:  # if neuron spiked
                            wi[x, q] = wi[x, q] + f_error * step*random.randint(80, 120)/100  # change weight strenght in proportion to error
                        q += 1
                x += 1
        i += 1

    print("Weights updated,  new weights:")
    print(wi)
    print(w)
    print(wo)
