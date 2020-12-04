import torch.nn as nn
import torch.nn.functional as F

# input is 1025x129

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 1025x129
    self.pool1 = nn.MaxPool2d((4,3), stride=3, padding = (1,0)) # 342 x 43
    self.conv2 = nn.Conv2d(32, 64, 3, padding = 1) # 342 x 43
    self.pool2 = nn.MaxPool2d(3, padding=(0,1)) # 114 x 15
    self.conv3 = nn.Conv2d(64, 128, 3, padding = 1) # 114 x 15
    self.pool3 = nn.MaxPool2d(3) # 38 x 5
    self.fc1 = nn.Linear(128 * 38 * 5, 32)
    self.fc2 = nn.Linear(32, 8)

  def forward(self, x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = self.pool3(F.relu(self.conv3(x)))
    x = x.view(-1, 128*38*5)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
   
