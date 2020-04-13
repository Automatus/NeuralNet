# NeuralNet v.2.1
# custom inputs, layers and outputs
# neurons are spiking neurons
# save and open files
# learning step depends on number of neurons and is randomized to avoid perfect parallelisms

import numpy as np
import random
import os

print("NeuralNet v3 by Automatus")

b = 0.5  # treshold/bias
step = 0.3  # learning rate
minrand = 500  # minimal number for randomization
maxrand = 1500  # maximal number for randomization

while True:
    # Showing available projects
    print("Choose your network:")
    networkslist = os.listdir(os.path.join(os.getcwd(), "Nets"))  # Reading folder with saved neural nets
    i = 1
    print("0. New Network")
    for net in networkslist:  # iterating trough files and printing them
        print(str(i) + ". " + net)
        i += 1
    choice = int(input()) - 1  # asking to choose a file
    if choice == -1:  # if new file: ask for all parameters
        ProjectChoice = input("Name of new network:\n")
        # inputs and hidden neurons and outputs and weights:
        number_of_inputs = int(input("number of inputs"))
        number_of_hidden_layers = int(input("number of hidden layers"))
        number_of_neurons_in_layer = int(input("number of neurons in each hidden layer"))
        number_of_outputs = int(input("number of outputs"))
        w = np.zeros((number_of_hidden_layers - 1, number_of_neurons_in_layer, number_of_neurons_in_layer))  # weights between neurons
        wi = np.zeros((number_of_inputs, number_of_neurons_in_layer))  # weights between inputs and first layer
        wo = np.zeros((number_of_neurons_in_layer, number_of_outputs))  # weights between last layer and outputs
        inputs = np.zeros((1, number_of_inputs))
        neurons = np.zeros((number_of_hidden_layers, number_of_neurons_in_layer))
        outputs = np.zeros((1, number_of_outputs))
        youtputs = np.zeros((1, number_of_outputs))  # desired outputs
        spll = np.zeros((1, number_of_neurons_in_layer))  # SPiking neurons in Last investigated Layer that have a connection to the output that is being updated

        variables = np.array([b, step, minrand, maxrand, number_of_inputs, number_of_hidden_layers, number_of_neurons_in_layer, number_of_outputs])  # This array is only created to be able to save it easily
        np.savez(os.path.join(os.getcwd(), "Nets", ProjectChoice), variables, wi, w, wo)
    else:  # if existing file:
        ProjectChoice = networkslist[choice]
        npzfile = np.load(os.path.join(os.getcwd(), "Nets", ProjectChoice))  # open file

        # Reading file
        b = npzfile["arr_0"][0]
        step = npzfile["arr_0"][1]
        minrand = npzfile["arr_0"][2]
        maxrand = npzfile["arr_0"][3]
        number_of_inputs = npzfile["arr_0"][4]
        number_of_hidden_layers = npzfile["arr_0"][5]
        number_of_neurons_in_layer = npzfile["arr_0"][6]
        number_of_outputs = npzfile["arr_0"][7]
        wi = npzfile["arr_1"]
        w = npzfile["arr_2"]
        wo = npzfile["arr_3"]
        
        inputs = np.zeros((1, number_of_inputs))
        neurons = np.zeros((number_of_hidden_layers, number_of_neurons_in_layer))
        outputs = np.zeros((1, number_of_outputs))
        youtputs = np.zeros((1, number_of_outputs))  # desired outputs
        spll = np.zeros((1, number_of_neurons_in_layer))  # SPiking neurons in Last investigated Layer that have a connection to the output that is being updated

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
            j = -2  # index for layer
            stop = False
            while j >= -number_of_hidden_layers-1:  # iterating through layers
                if not stop:
                    if j == -number_of_hidden_layers-1 and 1 in inputs:  # if all neurons are negative, but positive input:
                        t = 0  # index for input
                        for bla in inputs[0, :]:  # iterating through inputs
                            if bla:  # if input is active
                                r = 0
                                for thing in wi[t, :]:
                                    wi[t, r] = wi[t, r] + step*random.randint(minrand, maxrand)/1000  # strenghten weights/connections to neurons
                                    r += 1
                            t += 1
                    elif 1 in neurons[j, :]:  # if active neuron present in layer s
                        stop = True  # stop iterating through layers
                        t = 0  # index for neuron
                        for neuron in neurons[j, :]:  # iterating through neurons
                            if neuron:  # if neuron is active
                                r = 0
                                for thingy in w[j+1, :, t]:
                                    w[j+1, r, t] = w[j+1, r, t] + step/number_of_neurons_in_layer*random.randint(minrand, maxrand)/1000  # strenghten weights/connections in proportion to number of neurons
                                    r += 1
                            t += 1
                j += -1
        else:  # if active neurons in last layer are present
            f_error = youtputs[0, i] - outputs[0, i]  # calculate error
            j = -1  # index for layer of neurons
            k = 0  # index for neuron in layer j
            for neuron in neurons[j, :]:
                if neuron:  # if neuron is firing/1/True
                    wo[k, i] = wo[k, i] + f_error * step/number_of_neurons_in_layer*random.randint(minrand, maxrand)/1000  # change weight strenght in proportion to error and number of neurons
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
                                w[j+1, z, k] = w[j+1, z, k] + f_error * step/number_of_neurons_in_layer*random.randint(minrand, maxrand)/1000  # change weight strenght in proportion to error and number of neurons
                            z += 1
                    k += 1
                j += -1
            x = 0
            for innn in inputs[0, :]:  # iterating through inputs
                if innn:  # if input is 1
                    q = 0
                    for it in spll[0, :]:  # iterating through remembered spiking neurons
                        if spll[0, q]:  # if neuron spiked
                            wi[x, q] = wi[x, q] + f_error * step*random.randint(minrand, maxrand)/1000  # change weight strenght in proportion to error
                        q += 1
                x += 1
        i += 1

    print("Weights updated,  new weights:")
    print(wi)
    print(w)
    print(wo)
