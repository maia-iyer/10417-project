import torch
import torch.nn as nn
import torch.nn.functional as F

# input is 1025x129

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, (3,3),padding=(1,1)) 
    self.max1 = nn.MaxPool2d((3,1))
    self.avg1 = nn.AvgPool2d((1,3))
    self.conv2 = nn.Conv2d(32, 64, (3,3),padding=(1,1))
    self.max2 = nn.MaxPool2d((3,1))
    self.avg2 = nn.AvgPool2d((1,3))
    self.conv3 = nn.Conv2d(64, 128, (3,3),padding=(1,1))
    self.max3 = nn.MaxPool2d((3,1))
    self.avg3 = nn.AvgPool2d((1,3))
    self.fc1 = nn.Linear(128 * 37 * 4, 32)
    self.fc2 = nn.Linear(32, 8)

  def forward(self, x):
    x = self.avg1(self.max1(F.relu(self.conv1(x))))
    x = self.avg2(self.max2(F.relu(self.conv2(x))))
    x = self.avg3(self.max3(F.relu(self.conv3(x))))
    x = x.view(-1, 128 * 37 * 4)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x