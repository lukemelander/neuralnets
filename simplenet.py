import numpy as np
import random, os
lr = 1 #learning rate
bias = 1 #value of bias
weights = [random.random(), random.random(), random.random()] #weights

def Perceptron(input1,input2,output) :
    outputP = input1*weights[0]+input2*weights[1]+bias*weights[2]
    if outputP > 0 : #activation function, heavyside (either true or false, 0/1)
        outputP = 1
    else :
        outputP = 0
    error = output-outputP #Calculating error based on of expected answer
    weights[0] += error * input1 * lr #Recomputing weights
    weights[1] += error * input2 * lr
    weights[2] += error * bias * lr
print("Weights post training:")
print(str(weights[0])+' '+str(weights[1])+' '+str(weights[2]))
#Training the neural net to recognize a True or True input
for i in range(50) :
    if i==1 :
       print("Weights post training:")
       print(str(weights[0])+' '+str(weights[1])+' '+str(weights[2]))
    Perceptron(1,1,1) #True or true
    Perceptron(1,0,1) #True or false
    Perceptron(0,1,1) #False or true
    Perceptron(0,0,0) #False or False

#User input to test simple neural network
x = int(input())
y = int(input())
outputP = x*weights[0] + y*weights[1] + bias*weights[2]
if outputP > 0 :
    outputP = 1
else : 
    outputP = 0
print(x,"or",y,"is :",outputP)
