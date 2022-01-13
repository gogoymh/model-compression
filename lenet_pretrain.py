import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import os

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

train_loader = DataLoader(
                datasets.MNIST(
                        "../data/mnist",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.1307,), (0.3081,))]
                                ),
                        ),
                batch_size=128, shuffle=True)

test_loader = DataLoader(
                datasets.MNIST(
                        "../data/mnist",
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.1307,), (0.3081,))]
                                ),
                        ),
                batch_size=100, shuffle=False)

criterion = nn.CrossEntropyLoss()

model = LeNet()

x, y = train_loader.__iter__().next()

a = model(x)

'''
device = torch.device("cuda:0")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)

losses = torch.zeros((100))
for epoch in range(100):
    print("="*100)
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x.to(device))
        loss = criterion(output, y.to(device))
        loss.backward()
        optimizer.step()
        losses[epoch] += loss.item()
    losses[epoch] /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % (epoch, losses[epoch]), end="")
    
    accuracy = 0
    with torch.no_grad():
        model.eval()
        correct = 0
        for x, y in test_loader:
            output = model(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()                
        accuracy = correct / len(test_loader.dataset)
        print("[Accuracy:%f]" % accuracy)
        model.train()

torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses[epoch].item()}, "/home/super/ymh/modelcomp/results/Lenet5_Mnist.pth")
'''