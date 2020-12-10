import torch.nn as nn
import torch.nn.functional as F

# input is 1025x129

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, (3,3),padding=(1,1)) # 1025 x 32
    self.pool1 = nn.MaxPool2d((3,3)) # 341 x 32
    self.conv2 = nn.Conv2d(32, 64, (3,1),padding=(1,0)) # 341 x 32
    self.pool2 = nn.MaxPool2d((3,1)) # 113 x 32
    self.conv3 = nn.Conv2d(64, 128, (3,1),padding=(1,0)) # 113 x 32
    self.pool3 = nn.MaxPool2d((3,1)) # 37 x 32
    self.conv4 = nn.Conv2d(128, 256, (3,1),padding=(1,0))
    self.pool4 = nn.MaxPool2d((3,1))
    # self.fc1 = nn.Linear(256 * 12 * 43, 32)
    self.fc1 = nn.Linear(128 * 37 * 43, 32)
    self.fc2 = nn.Linear(32, 8)

  def forward(self, x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = self.pool3(F.relu(self.conv3(x)))
    # x = self.pool4(F.relu(self.conv4(x)))
    # x = x.view(-1, 256 * 12 * 43)
    x = x.view(-1, 128 * 37 * 43)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
