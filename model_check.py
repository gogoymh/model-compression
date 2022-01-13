import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import argparse
import timeit
import os

#################################################################################################################
from network import resnet18 as Net1
from network2 import ResNet18 as Net2
from utils import cal_sparsity, cal_ch, make_prob, cross_over, Mutate, minmax_fn


model1 = Net1()
model2 = Net2()

params1 = list(model1.parameters())
params2 = list(model2.parameters())

cnt1 = 0
cnt2 = 0
for i in range(62):
    cnt1 += params1[i].data.reshape(-1,).shape[0]
    cnt2 += params2[i].data.reshape(-1,).shape[0]
    
print("How many total parameters       | %d" % cnt1)
print("How many total parameters       | %d" % cnt2)

cnt1 = 0
cnt2 = 0
for i in range(62):
    if len(params1[i].data.shape) == 1:
        cnt1 += params1[i].data.reshape(-1,).shape[0]
        cnt2 += params2[i].data.reshape(-1,).shape[0]
        
print("How many 1-D parameters       | %d" % cnt1)
print("How many 1-D parameters       | %d" % cnt2)

cnt1 = 0
cnt2 = 0
for i in range(62):
    if len(params1[i].data.shape) == 2:
        cnt1 += params1[i].data.reshape(-1,).shape[0]
        cnt2 += params2[i].data.reshape(-1,).shape[0]
        
print("How many 2-D parameters       | %d" % cnt1)
print("How many 2-D parameters       | %d" % cnt2)

cnt1 = 0
cnt2 = 0
ch1 = 0
ch2 = 0
for i in range(62):
    if len(params1[i].data.shape) == 4:
        cnt1 += params1[i].data.reshape(-1,).shape[0]
        cnt2 += params2[i].data.reshape(-1,).shape[0]
        
        ch1 += params1[i].data.shape[0]
        ch2 += params2[i].data.shape[0]

print("How many 4-D parameters       | %d" % cnt1)
print("How many 4-D parameters       | %d" % cnt2)

print("How many channels             | %d" % ch1)
print("How many channels             | %d" % ch2)


train_loader = DataLoader(
                datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=False,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=128, shuffle=True, pin_memory = True)




















