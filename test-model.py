import os
import numpy as np
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from model import *
import torch.optim as optim
import time

print("Loading data...")
test_data = torch.load('va-data.pt')
testX = test_data['x']
testY = test_data['y']
numTest = testY.size()[0]
testInputs = torch.reshape(testX, (numTest, 1, 1025, 129))
testLabels = torch.reshape(testY, (-1,))
#print("Data loaded: ", numTrain, " training, ", numTest, " test, ")#, numValidation, " validation.")
print(numTest)

batchsize =  512
numBatches = numTest // batchsize 
numModels = 8

losses = []
accuracies = []
models = [0, 5, 10, 20, 30, 35, 40, 45]

for i in range(numModels):
  net = torch.load('models/modelc/model_avg_max_big' + str(models[i]) + '.pt')
  criterion = nn.CrossEntropyLoss()
  epoch_loss = 0.0
  epoch_training_accuracy = 0
  for batch in range(numBatches):
    if not batch == 0 and batch % 10 == 0: print(batch, numBatches)
    startInd = batch * batchsize
    endInd = batch * batchsize + 64
    inputs = torch.reshape(testX[startInd : endInd], (64, 1, 1025, 129))
    labels = torch.reshape(testY[startInd : endInd], (-1,))
    
    outputs = net(inputs)
    targets = labels.squeeze().long()
    loss = criterion(outputs, targets)

    epoch_loss += loss.item()
    output_answer = torch.argmax(outputs, 1)
    epoch_training_accuracy += torch.sum(output_answer == labels)
  train_acc = epoch_training_accuracy.item() / (numBatches * 64)
  epoch_loss /= numBatches
  losses.append(epoch_loss)
  accuracies.append(train_acc)
  print(models[i], epoch_loss, train_acc)

print(losses, accuracies)
 
