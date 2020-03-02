# NeuralNet, 2 inputls, 2 neural layers with each 3 neurons, 2 output neurons
# neurons are spiking neurons
# binary inputs and outputs
# learning algorithm compares desired output with actual output
# if the desired output is 0 and the actual output is 1 it searches back for spiking neurons and lowers the weights
# if the desired output is 1 and the actual output is 0 it searches back for spiking neurons and makes the weights higer
# or if there are no spiking neurons it searches for spiking neurons in earlier layers and strengthens them

#treshold or bias
b = 0.5

#learning rate
step = 0.1

#inputls and hidden neurons and outputsand weights
inputl = [0,0]
layer2 = [0,0,0]
layer2spike = [0,0,0]
layer3 = [0,0,0]
layer3spike = [0,0,0]
output = [0,0]
l2l1 = [0,0,0,0,0,0]
l3l2 = [0,0,0,0,0,0,0,0,0]
l4l3 = [0,0,0,0,0,0]

inputl[0] = int(input("input 1"))
inputl[1] = int(input("input 2"))

layer2[0] = inputl[0] * l2l1[0] + inputl[1] * l2l1[1]
if layer2[0] > b:
    layer2spike[0] = 1
else:
    layer2spike[0] = 0
layer2[1] = inputl[0] * l2l1[2] + inputl[1] * l2l1[3]
if layer2[1] > b:
    layer2spike[1] = 1
else:
    layer2spike[1] = 0
layer2[2] = inputl[0] * l2l1[4] + inputl[1] * l2l1[5]
if layer2[2] > b:
    layer2spike[2] = 1
else:
    layer2spike[2] = 0

layer3[0] = layer2spike[0] * l3l2[0] + layer2spike[1] * l3l2[1] + layer2spike[2] * l3l2[2]
if layer3[0] > b:
    layer3spike[0] = 1
else:
    layer3spike[0] = 0
layer3[1] = layer2spike[0] * l3l2[3] + layer2spike[1] * l3l2[4] + layer2spike[2] * l3l2[5]
if layer3[1] > b:
    layer3spike[1] = 1
else:
    layer3spike[1] = 0
layer3[2] = layer2spike[0] * l3l2[6] + layer2spike[1] * l3l2[7] + layer2spike[2] * l3l2[8]
if layer3[2] > b:
    layer3spike[2] = 1
else:
    layer3spike[2] = 0

output[0] = layer3spike[0] * l4l3[0] + layer3spike[1] * l4l3[1] + layer3spike[2] * l4l3[2]
output[1] = layer3spike[0] * l4l3[3] + layer3spike[1] * l4l3[4] + layer3spike[2] * l4l3[5]

print(output)

youtput0 = input("desired input 1")
youtput1 = input("desired input 2")

#learning with kind of backpropagation without derivatives (very bad code, I know)
if youtput0 == 0 and output[0] == 1:
    if layer3spike[0] == 1:
        l4l3[0] = l4l3[0] - step
        if layer2spike[0] == 1:
            l3l2[0] = l3l2[0] - step
            if inputl[0] == 1:
                l2l1[0] = l2l1[0] - step
            if inputl[1] == 1:
                l2l1[1] = l2l1[1] - step
        if layer2spike[1] == 1:
            l3l2[1] = l3l2[1] - step
            if inputl[0] == 1:
                l2l1[2] = l2l1[2] - step
            if inputl[1] == 1:
                l2l1[3] = l2l1[3] - step
        if layer2spike[2] == 1:
            l3l2[2] = l3l2[2] - step
            if inputl[0] == 1:
                l2l1[4] = l2l1[4] - step
            if inputl[1] == 1:
                l2l1[5] = l2l1[5] - step
    if layer3spike[1] == 1:
        l4l3[1] = l4l3[1] - step
        if layer2spike[0] == 1:
            l3l2[3] = l3l2[3] - step
            if inputl[0] == 1:
                l2l1[0] = l2l1[0] - step
            if inputl[1] == 1:
                l2l1[1] = l2l1[1] - step
        if layer2spike[1] == 1:
            l3l2[4] = l3l2[4] - step
            if inputl[0] == 1:
                l2l1[2] = l2l1[2] - step
            if inputl[1] == 1:
                l2l1[3] = l2l1[3] - step
        if layer2spike[2] == 1:
            l3l2[5] = l3l2[5] - step
            if inputl[0] == 1:
                l2l1[4] = l2l1[4] - step
            if inputl[1] == 1:
                l2l1[5] = l2l1[5] - step
    if layer3spike[2] == 1:
        l4l3[2] = l4l3[2] - step
        if layer2spike[0] == 1:
            l3l2[6] = l3l2[6] - step
            if inputl[0] == 1:
                l2l1[0] = l2l1[0] - step
            if inputl[1] == 1:
                l2l1[1] = l2l1[1] - step
        if layer2spike[1] == 1:
            l3l2[7] = l3l2[7] - step
            if inputl[0] == 1:
                l2l1[2] = l2l1[2] - step
            if inputl[1] == 1:
                l2l1[3] = l2l1[3] - step
        if layer2spike[2] == 1:
            l3l2[8] = l3l2[8] - step
            if inputl[0] == 1:
                l2l1[4] = l2l1[4] - step
            if inputl[1] == 1:
                l2l1[5] = l2l1[5] - step
if youtput1 == 0 and output[1] == 1:
    if layer3spike[0] == 1:
        l4l3[3] = l4l3[3] - step
        if layer2spike[0] == 1:
            l3l2[0] = l3l2[0] - step
            if inputl[0] == 1:
                l2l1[0] = l2l1[0] - step
            if inputl[1] == 1:
                l2l1[1] = l2l1[1] - step
        if layer2spike[1] == 1:
            l3l2[1] = l3l2[1] - step
            if inputl[0] == 1:
                l2l1[2] = l2l1[2] - step
            if inputl[1] == 1:
                l2l1[3] = l2l1[3] - step
        if layer2spike[2] == 1:
            l3l2[2] = l3l2[2] - step
            if inputl[0] == 1:
                l2l1[4] = l2l1[4] - step
            if inputl[1] == 1:
                l2l1[5] = l2l1[5] - step
    if layer3spike[1] == 1:
        l4l3[4] = l4l3[4] - step
        if layer2spike[0] == 1:
            l3l2[3] = l3l2[3] - step
            if inputl[0] == 1:
                l2l1[0] = l2l1[0] - step
            if inputl[1] == 1:
                l2l1[1] = l2l1[1] - step
        if layer2spike[1] == 1:
            l3l2[4] = l3l2[4] - step
            if inputl[0] == 1:
                l2l1[2] = l2l1[2] - step
            if inputl[1] == 1:
                l2l1[3] = l2l1[3] - step
        if layer2spike[2] == 1:
            l3l2[5] = l3l2[5] - step
            if inputl[0] == 1:
                l2l1[4] = l2l1[4] - step
            if inputl[1] == 1:
                l2l1[5] = l2l1[5] - step
    if layer3spike[2] == 1:
        l4l3[5] = l4l3[5] - step
        if layer2spike[0] == 1:
            l3l2[6] = l3l2[6] - step
            if inputl[0] == 1:
                l2l1[0] = l2l1[0] - step
            if inputl[1] == 1:
                l2l1[1] = l2l1[1] - step
        if layer2spike[1] == 1:
            l3l2[7] = l3l2[7] - step
            if inputl[0] == 1:
                l2l1[2] = l2l1[2] - step
            if inputl[1] == 1:
                l2l1[3] = l2l1[3] - step
        if layer2spike[2] == 1:
            l3l2[8] = l3l2[8] - step
            if inputl[0] == 1:
                l2l1[4] = l2l1[4] - step
            if inputl[1] == 1:
                l2l1[5] = l2l1[5] - step
if youtput0 == 1 and output[0] == 0:
    if layer3spike[0] == 1:
        l4l3[0] = l4l3[0] + step
        if layer2spike[0] == 1:
            l3l2[0] = l3l2[0] + step
            if inputl[0] == 1:
                l2l1[0] = l2l1[0] + step
            if inputl[1] == 1:
                l2l1[1] = l2l1[1] + step
        if layer2spike[1] == 1:
            l3l2[1] = l3l2[1] + step
            if inputl[0] == 1:
                l2l1[2] = l2l1[2] + step
            if inputl[1] == 1:
                l2l1[3] = l2l1[3] + step
        if layer2spike[2] == 1:
            l3l2[2] = l3l2[2] + step
            if inputl[0] == 1:
                l2l1[4] = l2l1[4] + step
            if inputl[1] == 1:
                l2l1[5] = l2l1[5] + step
    if layer3spike[1] == 1:
        l4l3[1] = l4l3[1] + step
        if layer2spike[0] == 1:
            l3l2[3] = l3l2[3] + step
            if inputl[0] == 1:
                l2l1[0] = l2l1[0] + step
            if inputl[1] == 1:
                l2l1[1] = l2l1[1] + step
        if layer2spike[1] == 1:
            l3l2[4] = l3l2[4] + step
            if inputl[0] == 1:
                l2l1[2] = l2l1[2] + step
            if inputl[1] == 1:
                l2l1[3] = l2l1[3] + step
        if layer2spike[2] == 1:
            l3l2[5] = l3l2[5] + step
            if inputl[0] == 1:
                l2l1[4] = l2l1[4] + step
            if inputl[1] == 1:
                l2l1[5] = l2l1[5] + step
    if layer3spike[2] == 1:
        l4l3[2] = l4l3[2] + step
        if layer2spike[0] == 1:
            l3l2[6] = l3l2[6] + step
            if inputl[0] == 1:
                l2l1[0] = l2l1[0] + step
            if inputl[1] == 1:
                l2l1[1] = l2l1[1] + step
        if layer2spike[1] == 1:
            l3l2[7] = l3l2[7] + step
            if inputl[0] == 1:
                l2l1[2] = l2l1[2] + step
            if inputl[1] == 1:
                l2l1[3] = l2l1[3] + step
        if layer2spike[2] == 1:
            l3l2[8] = l3l2[8] - step
            if inputl[0] == 1:
                l2l1[4] = l2l1[4] + step
            if inputl[1] == 1:
                l2l1[5] = l2l1[5] + step
    elif True:
        #here code to search for firing neurons in layer2 and strengthen weights before and after it
if youtput1 == 1 and output[1] == 0:
    if layer3spike[0] == 1:
        l4l3[3] = l4l3[3] + step
        if layer2spike[0] == 1:
            l3l2[0] = l3l2[0] + step
            if inputl[0] == 1:
                l2l1[0] = l2l1[0] + step
            if inputl[1] == 1:
                l2l1[1] = l2l1[1] + step
        if layer2spike[1] == 1:
            l3l2[1] = l3l2[1] + step
            if inputl[0] == 1:
                l2l1[2] = l2l1[2] + step
            if inputl[1] == 1:
                l2l1[3] = l2l1[3] + step
        if layer2spike[2] == 1:
            l3l2[2] = l3l2[2] + step
            if inputl[0] == 1:
                l2l1[4] = l2l1[4] + step
            if inputl[1] == 1:
                l2l1[5] = l2l1[5] + step
    if layer3spike[1] == 1:
        l4l3[4] = l4l3[4] + step
        if layer2spike[0] == 1:
            l3l2[3] = l3l2[3] + step
            if inputl[0] == 1:
                l2l1[0] = l2l1[0] + step
            if inputl[1] == 1:
                l2l1[1] = l2l1[1] + step
        if layer2spike[1] == 1:
            l3l2[4] = l3l2[4] + step
            if inputl[0] == 1:
                l2l1[2] = l2l1[2] + step
            if inputl[1] == 1:
                l2l1[3] = l2l1[3] + step
        if layer2spike[2] == 1:
            l3l2[5] = l3l2[5] + step
            if inputl[0] == 1:
                l2l1[4] = l2l1[4] + step
            if inputl[1] == 1:
                l2l1[5] = l2l1[5] + step
    if layer3spike[2] == 1:
        l4l3[5] = l4l3[5] + step
        if layer2spike[0] == 1:
            l3l2[6] = l3l2[6] + step
            if inputl[0] == 1:
                l2l1[0] = l2l1[0] + step
            if inputl[1] == 1:
                l2l1[1] = l2l1[1] + step
        if layer2spike[1] == 1:
            l3l2[7] = l3l2[7] + step
            if inputl[0] == 1:
                l2l1[2] = l2l1[2] + step
            if inputl[1] == 1:
                l2l1[3] = l2l1[3] + step
        if layer2spike[2] == 1:
            l3l2[8] = l3l2[8] + step
            if inputl[0] == 1:
                l2l1[4] = l2l1[4] + step
            if inputl[1] == 1:
                l2l1[5] = l2l1[5] + step
    elif True:
        #here code to search for firing neurons in layer2 and strengthen weights before and after it
