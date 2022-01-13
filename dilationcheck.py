import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import argparse
import timeit
import os
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    
    self.conv1 = nn.Conv2d(1, 6, 5) # 0
    self.conv3 = nn.Conv2d(6, 16, 5) # 2
    self.conv5 = nn.Conv2d(16, 120, 5) # 4
    
    self.fc6 = nn.Linear(120, 84) # 6
    self.fc7 = nn.Linear(84, 10) # 8
    
  def forward(self, x):
      #print(x.shape)
      x = F.max_pool2d(F.relu(self.conv1(x)), 2)
      #print(x.shape)
      x = F.max_pool2d(F.relu(self.conv3(x)), 2)
      #print(x.shape)
      x = F.relu(self.conv5(x))
      #print(x.shape)
      x = x.view(-1, 120)
      #print(x.shape)
      x = F.relu(self.fc6(x))
      #print(x.shape)
      x = self.fc7(x)
      #print(x.shape)
      return F.log_softmax(x, dim=0)
  
class dilation(nn.Module):
    def __init__(self):
        super(dilation, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, (3,2), 1, 0, (2,4))
        self.conv3 = nn.Conv2d(6, 16, (3,2), 1, 0, (2,4))
        self.conv5 = nn.Conv2d(16, 120, (3,2), 1, 0, (2,4))
        
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, 10)
    
    def forward(self, x):
        print(x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        print(x.shape)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        print(x.shape)
        x = F.relu(self.conv5(x))
        print(x.shape)
        x = x.view(-1, 120)
        print(x.shape)
        x = F.relu(self.fc6(x))
        print(x.shape)
        x = self.fc7(x)
        print(x.shape)
        return F.log_softmax(x, dim=0)
    
class dilation2(nn.Module):
    def __init__(self):
        super(dilation2, self).__init__()
        
        self.conv1_1 = nn.Conv2d(1, 6, (1,2), 1, 0)
        self.conv1_2 = nn.Conv2d(1, 6, (2,2), 1, 0)
        
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.conv5 = nn.Conv2d(16, 120, 5)
        
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, 10)
    
    def forward(self, x):
        print(x.shape)
        
        x_1 = self.conv1_1(x[:,:,:28,:29])
        x_2 = self.conv1_2(x[:,:,2:31,2:31])
        print("x_1", x_1.shape)
        print("x_2", x_2.shape)
        
        x = x_1 + x_2
        print("x_1 + x_2", x.shape)
        
        x = F.max_pool2d(F.relu(x), 2)
        print(x.shape)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        print(x.shape)
        x = F.relu(self.conv5(x))
        print(x.shape)
        x = x.view(-1, 120)
        print(x.shape)
        x = F.relu(self.fc6(x))
        print(x.shape)
        x = self.fc7(x)
        print(x.shape)
        return F.log_softmax(x, dim=0)    

class dilation3(nn.Module):
    def __init__(self):
        super(dilation3, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, (3,2), (3,2), (26,12))
        self.conv3 = nn.Conv2d(6, 16, (3,2), 1, 0)
        self.conv5 = nn.Conv2d(16, 120, (3,2), 1, 0)
        
        self.fc6 = nn.Linear(120*4*6, 84)
        self.fc7 = nn.Linear(84, 10)
    
    def forward(self, x):
        print(x.shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        print(x.shape)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        print(x.shape)
        x = F.relu(self.conv5(x))
        print(x.shape)
        x = x.view(128, -1)
        print(x.shape)
        x = F.relu(self.fc6(x))
        print(x.shape)
        x = self.fc7(x)
        print(x.shape)
        return F.log_softmax(x, dim=0)


train_loader = DataLoader(
                datasets.MNIST(
                        "../data/mnist",
                        train=True,
                        download=False,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.1307,), (0.3081,))]
                                ),
                        ),
                batch_size=128, shuffle=True, pin_memory = True)

x, y = train_loader.__iter__().next()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = dilation2().to(device)

output = model(x.to(device))






   
    
    