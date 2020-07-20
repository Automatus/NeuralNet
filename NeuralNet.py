#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NeuralNet v.3, MODULE by Automatus.

neurons are spiking neurons.
learning step depends on number of neurons and is randomized.
to avoid perfect parallelisms.
"""

import numpy as np
import random
import os
from gtts import gTTS


def new():
    """Create new neural network with command line."""
    print("NeuralNet version 3 by Automatus: Creating new Neural Network...")

    if not os.path.exists(os.path.join(os.getcwd(), "Nets")):
        os.makedirs(os.path.join(os.getcwd(), "Nets"))

    project_choice = input("Name of new network:\n")

    b = float(input("Treshold value / bias, for example 0.5"))
    step = float(input("Step size for learning rate, for example 0.3"))
    minrand = int(input("minimum number for randomization, for example 500"))
    maxrand = int(input("maximum number for randomization, for example 1500"))
    negative_error_scalar = float(input("scalar for negative error, for example 10"))
    number_of_inputs = int(input("number of inputs"))
    number_of_hidden_layers = int(input("number of hidden layers"))
    number_of_neurons_in_layer = int(input("number of neurons in each hidden layer"))
    number_of_outputs = int(input("number of outputs"))

    w = np.zeros((number_of_hidden_layers - 1, number_of_neurons_in_layer, number_of_neurons_in_layer))
    # weights between neurons
    wi = np.zeros((number_of_inputs, number_of_neurons_in_layer))
    # weights between inputs and first layer
    wo = np.zeros((number_of_neurons_in_layer, number_of_outputs))
    # weights between last layer and outputs

    variables = np.array([b, step, minrand, maxrand, number_of_inputs, number_of_hidden_layers, number_of_neurons_in_layer, number_of_outputs, negative_error_scalar])
    # This array is only created to be able to save it easily

    np.savez(os.path.join(os.getcwd(), "Nets", project_choice), variables, wi, w, wo)
    print("Neural Network generated and saved")


def calc(file, inputarray):
    """
    Calculate output np array for a given neural network
    and input np.array with format (1,x)."
    """
    print("NeuralNet version 3 by Automatus: Calculating answer...")

    npzfile = np.load(file)  # open file
    # Reading file:
    b = npzfile["arr_0"][0]
    number_of_hidden_layers = int(npzfile["arr_0"][5])
    number_of_neurons_in_layer = int(npzfile["arr_0"][6])
    number_of_outputs = int(npzfile["arr_0"][7])
    wi = npzfile["arr_1"]
    w = npzfile["arr_2"]
    wo = npzfile["arr_3"]

    neurons = np.zeros((number_of_hidden_layers, number_of_neurons_in_layer), dtype=bool)
    outputs = np.zeros((1, number_of_outputs))

    i = 0  # = current layer
    while i <= number_of_hidden_layers:  # iterating through layers
        if i == 0:  # if calculating firts layer
            neurons[i, :] = (inputarray.dot(wi)) > b
        elif i == number_of_hidden_layers:  # if calculating last layer
            outputs = neurons[number_of_hidden_layers - 1, :].reshape(1, number_of_neurons_in_layer).dot(wo)
            # dot product of last neural layer with weigts for output
        else:  # if calculating layers in between
            neurons[i, :] = (neurons[i - 1, :].dot(w[i - 1, :])) > b
        i += 1

    print("NeuralNet version 3 by Automatus: Answer calculated")
    return outputs


def autolearn(file, datafolder):
    """
    Execute Learning Algorithm.

    File = neural network file path
    first array represents input in format (1, x)
    second array represents desired output values in format (1, x)
    """
    import matplotlib.pyplot as plt

    def plotit(values):
        fig, ax = plt.subplots()

        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        # https://www.science-emergence.com/Articles/How-to-change-the-color-background-of-a-matplotlib-figure-/
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(axis='x', colors='yellow')
        ax.tick_params(axis='y', colors='yellow')
        # https://stackoverflow.com/questions/1982770/matplotlib-changing-the-color-of-an-axis#12059429
        ax.set_xlabel("iterations", color="yellow")
        ax.set_ylabel("error", color="yellow")
        ax.set_title("Error for each try", color="yellow")

        ax.plot(list(range(len(values))), values)
        # list range: https://stackoverflow.com/questions/11480042/python-3-turn-range-to-a-list
        plt.show()
        plt.close()

    print("NeuralNet v3 by Automatus: Auto Teaching Neural Network...")
    repeat = int(input("How manu times should the proces be repeated? (for example 1)"))

    errorvalues = []

    npzfile = np.load(file)  # open file

    # Reading file
    b = npzfile["arr_0"][0]
    step = npzfile["arr_0"][1]
    minrand = npzfile["arr_0"][2]
    maxrand = npzfile["arr_0"][3]
    number_of_inputs = int(npzfile["arr_0"][4])
    number_of_hidden_layers = int(npzfile["arr_0"][5])
    number_of_neurons_in_layer = int(npzfile["arr_0"][6])
    number_of_outputs = int(npzfile["arr_0"][7])
    negative_error_scalar = npzfile["arr_0"][8]
    wi = npzfile["arr_1"]
    w = npzfile["arr_2"]
    wo = npzfile["arr_3"]

    print("Current learning variables:")
    print("bias/treshold                  = " + str(b))
    print("stepsize/learning rate         = " + str(step))
    print("lower number for randomization = " + str(minrand))
    print("upper number for randomization = " + str(maxrand))
    print("scalar for negative error      = " + str(negative_error_scalar))

    print("Do you want to change the learning variables of the network? y/n")
    decide = input("(The new variables will be saved in the networks file)")
    if decide == "y":
        b = float(input(
            "Treshold value / bias, for example 0.5"))
        step = float(input(
            "Step size for learning rate, for example 0.3"))
        minrand = int(input(
            "minimum number for randomization, for example 500"))
        maxrand = int(input(
            "maximum number for randomization, for example 1500"))
        negative_error_scalar = float(input(
            "scalar for negative error, for example 10"))
        print("Variables Changed")

    neurons = np.zeros((number_of_hidden_layers, number_of_neurons_in_layer), dtype=bool)
    outputs = np.zeros((1, number_of_outputs))
    spll = np.zeros((1, number_of_neurons_in_layer), dtype=bool)
    # SPiking neurons in Last investigated Layer that have a
    # connection to the output that is being updated
    randomvalues = np.zeros((1, number_of_neurons_in_layer))

    variables = np.array([b, step, minrand, maxrand, number_of_inputs, number_of_hidden_layers, number_of_neurons_in_layer, number_of_outputs, negative_error_scalar])

    objectimgpaths = [f for f in os.listdir(datafolder) if os.path.isfile(os.path.join(datafolder, f))]

    repeating = 0
    while repeating < repeat:
        running = 0
        while running < len(objectimgpaths):
            npzdatafile = np.load(os.path.join(datafolder, objectimgpaths[running]))
            print(objectimgpaths[running])
            inputs = npzdatafile["arr_0"]
            youtputs = npzdatafile["arr_1"]

            # Calculating the output
            print("calculating answer")
            print("desired answer:   ", (youtputs > 0.5)*1)
            i = 0  # = current layer
            while i <= number_of_hidden_layers:  # iterating through layers
                if i == 0:  # if calculating firts layer
                    neurons[i, :] = (inputs.dot(wi)) > b
                    # dot product of inputs with weights for inputs and
                    # first layer, checks if it is bigger then treshold,
                    # then sets neuron as firing if true
                elif i == number_of_hidden_layers:  # if calculating last layer
                    outputs = neurons[number_of_hidden_layers - 1, :].reshape(1, number_of_neurons_in_layer).dot(wo)
                    # dot product of last neural layer with weigts for output
                else:  # if not calculating layers in between
                    neurons[i, :] = (neurons[i - 1, :].dot(w[i - 1, :])) > b
                    # dot product of current neural layer with weigths, checks
                    # if it is bigger then treshold, then sets neuron
                    # as firing if true
                i += 1
            print("final answer:     ", (outputs > 0.5)*1)
            print("calculated answer:", outputs)

            #  learning algorithm
            i = 0  # index for output
            for item in youtputs[0, :]:  # iterating through outputs
                f_error = youtputs[0, i] - outputs[0, i]  # calculate error
#                errorvalues.append(f_error)
#                plotit(errorvalues)
                if 1 not in neurons[-1, :] and item:
                    # if no active neurons in last layer and desired output is 1
                    plo = 0
                    for randomvalue in randomvalues[0, :]:
                        #  setting random values
                        randomvalues[0, plo] = random.randint(minrand, maxrand) / 1000
                        plo += 1
                    j = -2  # index for layer
                    stop = False
                    while j >= -number_of_hidden_layers - 1:
                        # iterating through layers
                        if not stop:
                            if j == (-number_of_hidden_layers - 1) and np.any(e > 0 for e in inputs):
                                # if all neurons are negative, but positive input:
                                t = 0  # index for input
                                for bla in inputs[0, :]:  # iterating inputs
                                    if bla > 0:  # if input is active
                                        r = 0
                                        for thing in wi[t, :]:
                                            wi[t, r] = (wi[t, r] + step / number_of_inputs * bla * randomvalues[0, r])
                                            # strenghten weights to neurons
                                            r += 1
                                    t += 1
                            elif 1 in neurons[j, :] and j is not (-number_of_hidden_layers - 1):
                                # if active neuron present in layer j
                                stop = True
                                t = 0  # index for neuron
                                for neuron in neurons[j, :]:  # iterating neurons
                                    if neuron:  # if neuron is active
                                        r = 0
                                        for thingy in w[j + 1, :, t]:
                                            w[j + 1, r, t] = w[j + 1, r, t] + (step / number_of_neurons_in_layer * randomvalues[0, r])
                                            # strenghten weights in proportion to
                                            # number of neurons
                                            r += 1
                                    t += 1
                        j += -1
                elif not youtputs[0, i] == outputs[0, i]:
                    # if active neurons in last layer are present
                    # (and error  is not 0)
                    if f_error < 0:
                        f_error = f_error * negative_error_scalar
                        # without this it stays around 0.5 as output,
                        # this clears the way for new connections to be formed
                    j = -1  # index for layer of neurons
                    k = 0  # index for neuron in layer j
                    for neuron in neurons[j, :]:
                        if neuron:  # if neuron is firing
                            wo[k, i] = wo[k, i] + f_error * (step / number_of_neurons_in_layer)
                            # change weight strenght in proportion to
                            # error and number of neurons
                            spll[0, k] = True
                            # remember spiking neuron for next iterations
                        k += 1
                    j = j - 1
                    while j > -number_of_hidden_layers:
                        # while iterating through neural layers
                        k = 0
                        for neuron in neurons[j, :]:
                            # iterating through neurons in layer j
                            if neuron:  # if neuron fired
                                z = 0
                                for thing in spll[0, :]:
                                    # iterating through remembered spiking neurons
                                    if spll[0, z]:  # if neuron spiked
                                        w[j + 1, z, k] = w[j + 1, z, k] + (f_error * step / number_of_neurons_in_layer)
                                        # change weight strenght in proportion to
                                        # error and number of neurons
                                    z += 1
                            k += 1
                        j += -1
                    x = 0
                    for innn in inputs[0, :]:  # iterating through inputs
                        if innn > 0:  # if input is 1
                            q = 0
                            for it in spll[0, :]:
                                # iterating through remembered spiking neurons
                                if spll[0, q]:  # if neuron spiked
                                    wi[x, q] = wi[x, q] + f_error * innn * step
                                    # change weight strenght in proportion to
                                    # error and input signal strength
                                q += 1
                        x += 1
                i += 1
            running += 1
        repeating += 1

    np.savez(file, variables, wi, w, wo)
    print("NeuralNet v3 by Automatus: Weights updated\n")
    tts = gTTS(text='Neural Networks weights updated and saved', lang='en')
    tts.save("status.wav")
    os.system("mpg321 status.wav")


def imgtodata():
    """Create Data with webcam."""
    import cv2

    print("NeuralNet by Automatus: Making Data...")

    if not os.path.exists(os.path.join(os.getcwd(), "Data")):
        os.makedirs(os.path.join(os.getcwd(), "Data"))

    cv2.namedWindow("preview")
    # probably got it from here: https://stackoverflow.com/questions/19285562/python-opencv-imread-displaying-image

    youtputs = np.zeros((1, 1))

    name = input("Name for data entry:")

    os.mkdir((os.path.join(os.getcwd(), "Data", name)))

    currentimage = cv2.VideoCapture(0)
    # maybe got it from here: https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
    currentimage.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv-python
    succes, currentframelll = currentimage.read()

    for running in range(30):
        cv2.waitKey(1000)

        if running == 0:
            input("Please make the object APPEAR in the next photos")
            youtputs[0, 0] = 1
        if running == 15:
            input("Please make NO(!) object appear in the next photos")
            youtputs[0, 0] = 0
            succes, currentframelll = currentimage.read()

        succes, currentframelll = currentimage.read()
        cv2.imshow("preview", currentframelll)
        # probably got it from here: https://stackoverflow.com/questions/19285562/python-opencv-imread-displaying-image
        resizedd = cv2.resize(currentframelll, None, fx=0.1, fy=0.1,
                              interpolation=cv2.INTER_LINEAR)
        red = resizedd[:, :, 0]
        # + currentframe[:, :, 1] + currentframe[:, :, 2]
        c = np.reshape(red, (1, 3072))
        inputs = c / 255
        inputs = inputs > 0.4
        # https://stackoverflow.com/questions/36719997/threshold-in-2d-numpy-array#36720130
        filename = name + str(running) + ".npz"
        np.savez((os.path.join(os.getcwd(), "Data", name, filename)), inputs, youtputs)

        print(running)

    cv2.destroyAllWindows()
    print("array shape = ", inputs.shape)


def resetnet(file):
    """Reset network."""
    print("Loading file to reset...")
    npzfile = np.load(file)

    b = npzfile["arr_0"][0]
    step = npzfile["arr_0"][1]
    minrand = npzfile["arr_0"][2]
    maxrand = npzfile["arr_0"][3]
    number_of_inputs = int(npzfile["arr_0"][4])
    number_of_hidden_layers = int(npzfile["arr_0"][5])
    number_of_neurons_in_layer = int(npzfile["arr_0"][6])
    number_of_outputs = int(npzfile["arr_0"][7])
    negative_error_scalar = npzfile["arr_0"][8]
    wi = npzfile["arr_1"]
    w = npzfile["arr_2"]
    wo = npzfile["arr_3"]
    variables = np.array([b, step, minrand, maxrand, number_of_inputs, number_of_hidden_layers, number_of_neurons_in_layer, number_of_outputs, negative_error_scalar])
    value = input("New value (for example 0):")
    print("Resetting...")
    wi[:] = float(value)
    w[:] = float(value)
    wo[:] = float(value)
    print("saving...")
    np.savez(file, variables, wi, w, wo)
    print("File resetted and saved")


def mnist():
    """Convert MNIST keras numpy arrays to usable data for NeuralNet."""
    print("NeuralNet by Automatus: Making MNIST Data...")

    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    if not os.path.exists(os.path.join(os.getcwd(), "Data")):
        os.makedirs(os.path.join(os.getcwd(), "Data"))

    youtputs = np.zeros((1, 10))
    inputs = np.zeros((1, 28 * 28))

    name = input("Name for data entry:")
    nametest = input("Name for TEST data entry:")

    os.mkdir((os.path.join(os.getcwd(), "Data", name)))
    os.mkdir((os.path.join(os.getcwd(), "Data", nametest)))

    for running in range(len(train_labels)):
        inputs = train_images[running, :, :].reshape((1, 28*28))
        youtputs[:] = 0
        youtputs[0, (train_labels[running])] = 1
        filename = name + str(running) + ".npz"
        np.savez((os.path.join(os.getcwd(), "Data", name, filename)), inputs, youtputs)
    print("training data converted and saved...")

    for running in range(len(test_labels)):
        inputs = test_images[running, :, :].reshape((1, 28*28))
        youtputs[:] = 0
        youtputs[0, (test_labels[running])] = 1
        filename = nametest + str(running) + ".npz"
        np.savez((os.path.join(os.getcwd(), "Data", nametest, filename)), inputs, youtputs)
    print("test data converted and saved.")


tts = gTTS(text='Welcome to NeuralNet by Automatus', lang='en')
tts.save("status.mp3")
os.system("mpg321 status.mp3")
# to eliminate delayed playback look at:https://askubuntu.com/questions/218444/sound-output-starts-delayed?newreg=de54ca755d054ccb98a6c7a4a0f84c5d
execute = True
filechosen = False
while execute:
    print("\nNeuralNet v3 by Automatus: Module\n\n")
    print("Choose a function. Options:")
    print("ENTER.   Continue with current network and function")
    print("0.       New Network")
    print("1.       Calculate output with Network")
    print("2.       Reset Network")
    print("3.       Auto teaching of Network")
    print("4.       Take photos with webcam to make training/test data")
    print("5.       Make data from MNIST data")
    print("q.       Quit ")

    beslis = input()

    if beslis != "":
        last = beslis
        filechosen = False
    else:
        beslis = last
        filechosen = True

    if beslis == "1" or beslis == "3" or beslis == "2":
        if not filechosen:
            if not os.path.exists(os.path.join(os.getcwd(), "Nets")):
                print("No neural Networks found, create one with option 0.")
                continue
            netlist = os.listdir(os.path.join(os.getcwd(), "Nets"))
            i = 0
            for net in netlist:
                print(str(i) + ".        " + net)
                i += 1
            netchoosen = int(input("Choose Network"))
            thisfile = os.path.join(os.getcwd(), "Nets", netlist[netchoosen])

    if beslis == "q":
        execute = False
    elif beslis == "0":
        new()
    elif beslis == "1":
        if not os.path.exists(os.path.join(os.getcwd(), "Data")):
            print("No data to procces found, create data with option 4.")
            continue
        netlist = os.listdir(os.path.join(os.getcwd(), "Data"))
        i = 0
        for net in netlist:
            print(str(i) + ".        " + net)
            i += 1
        print("Choose Datafolder")
        netchoosen = int(input())
        thisfolder = os.path.join(os.getcwd(), "Data", netlist[netchoosen])
        netlist = os.listdir(thisfolder)
        i = 0
        for net in netlist:
            print(str(i) + ".        " + net)
            i += 1
        print("Choose item")
        netchoosen = int(input())
        thisdata = os.path.join(os.getcwd(), "Data", thisfolder,
                                netlist[netchoosen])
        thisarray = np.load(thisdata)
        thisitem = thisarray["arr_0"]
        print(thisfile)
        print(thisitem)
        answer = calc(thisfile, thisitem)
        print("answer = ", answer)
    elif beslis == "2":
        print("Are you sure you want to reset " + thisfile + "? y/n")
        yesno = input()
        if yesno == "y":
            resetnet(thisfile)
    elif beslis == "3":
        if not os.path.exists(os.path.join(os.getcwd(), "Data")):
            print("No data to procces found, create data with option 4.")
            continue
        netlist = os.listdir(os.path.join(os.getcwd(), "Data"))
        i = 0
        for net in netlist:
            print(str(i) + ".        " + net)
            i += 1
        print("Choose Datafolder")
        netchoosen = int(input())
        thisdata = os.path.join(os.getcwd(), "Data", netlist[netchoosen])
        autolearn(thisfile, thisdata)
    elif beslis == "4":
        imgtodata()
    elif beslis == "5":
        mnist()
