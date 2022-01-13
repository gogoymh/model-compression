import torch.nn as nn
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets

from network import resnet18 as ResNet


test_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
                        download=False,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=100, shuffle=False, pin_memory=True)
                                
model = ResNet().cuda()
checkpoint = torch.load("C:/유민형/개인 연구/model compression/models/resnet18-5c106cde.pth")
model.load_state_dict(checkpoint)
criterion = nn.CrossEntropyLoss()

correct = 0
loss = 0
for x,y in test_loader:
    output = model(x.float().cuda())
    loss_tmp = criterion(output, y.long().cuda())
    pred = output.argmax(1, keepdim=True)
    correct += pred.eq(y.long().cuda().view_as(pred)).sum().item()
    loss += loss_tmp.item()

loss /= len(test_loader)
accuracy = correct/len(test_loader.dataset)
print(loss)
print(accuracy)