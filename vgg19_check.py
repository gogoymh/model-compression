# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:12:12 2019

@author: Minhyeong
"""

import torch
import torchvision.models as models
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets

device = 'cuda:0'

model = models.vgg19_bn(num_classes=10)
model.to(device)
checkpoint = torch.load("C://results//vgg19_bn.pt", map_location=device)
model.load_state_dict(checkpoint)

test_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=100, shuffle=False, pin_memory=True)

correct = 0
for x, y in test_loader:
    output = model(x.float().to(device))
    pred = output.argmax(1, keepdim=True)
    correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
    accuracy = correct / len(test_loader.dataset)
print("[Accuracy:%f]" % accuracy)