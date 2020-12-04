import os
import numpy as np
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from model import *
import torch.optim as optim
import time

print("Creating net...")
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
torch.save(net, 'model.pt')
print("Loading data...")
training_data = torch.load('te-data.pt')
trainX = training_data['x']
trainY = training_data['y']
numTrain = trainY.size()[0]
trainClassIndices = [0, 251, 656, 974, 1129, 3530, 3653, 3802, 5841] 
testClassIndices = [0, 733, 1688, 2830, 5997, 8606, 8931, 9337, 12236]
test_data = torch.load('va-data.pt')
testX = test_data['x']
testY = test_data['y']
numTest = testY.size()[0]
testInputs = torch.reshape(testX, (numTest, 1, 1025, 129))
testLabels = torch.reshape(testY, (-1,))
"""
validation_data = torch.load('va-data.pt')
validationX = validation_data['x']
validationY = validation_data['y']
numValidation = validationY.size()[0]"""

#numTest = np.shape(testY)
#numValidation = np.shape(validationY)
print("Data loaded: ", numTrain, " training, ", numTest, " test, ")#, numValidation, " validation.")

numEpochs = 2
batchsize =  32
numBatches = numTrain // batchsize

for epoch in range(numEpochs):
  epochStart = time.time()
  epoch_loss = 0.0
  epoch_training_accuracy = 0
  for batch in range(numBatches):
    batchStart = time.time()
    classChoices = torch.randint(8, (batchsize,))
    indices = torch.tensor([torch.randint(testClassIndices[c], testClassIndices[c+1], (1,)) for c in classChoices])
    inputs = torch.reshape(trainX[indices], (32, 1, 1025, 129))
    labels = torch.reshape(trainY[indices], (-1,))
    optimizer.zero_grad()
    
    outputs = net(inputs)
    targets = labels.squeeze().long()
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    output_answer = torch.argmax(outputs, 1)
    epoch_training_accuracy += torch.sum(output_answer == labels)
    if batch % 30 == 0: 
      print("minibatch ", batch, " out of ", numBatches, " with loss ", loss.item())
    batchEnd = time.time()
    print("Epoch time taken is ", batchEnd - batchStart)
  torch.save(net, 'models/model'+ str(epoch) +'.pt')
  train_acc = epoch_training_accuracy.item() / numTrain
  # test loss and test acc
  randomTestIndices = torch.randint(numTest, (128,))
  testOutputs = net(testInputs[randomTestIndices])
  testLoss = criterion(testOutputs, testLabels.squeeze().long()[randomTestIndices]).item()
  testAnswer = torch.argmax(testOutputs, 1)
  testAcc = torch.sum(testAnswer == testLabels[randomTestIndices]).item() / 128
  epochEnd = time.time()
  print("Epoch time taken is ", epochEnd - epochStart)
  
  print("Epoch ", epoch, " has train loss ", epoch_loss/numBatches, " and accuracy ", train_acc)
  print("Epoch ", epoch, " has test loss ", testLoss, " and accuracy ", testAcc)
