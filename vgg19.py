# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:29:08 2019

@author: Minhyeong
"""
#import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
#import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
import math

##########################################################################################################################
class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#########################################################################################################################
def _vgg(arch, cfg, batch_norm, progress, **kwargs):

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    return model

def vgg19(progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, progress, **kwargs)


def vgg19_bn(progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, progress, **kwargs)

##########################################################################################################################

train_loader = DataLoader(
                datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=128, shuffle=True, pin_memory=True)

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

criterion = nn.CrossEntropyLoss()

model = vgg19()#vgg19_bn(num_classes=10)
print(model)
print("="*100)

device = torch.device("cuda:0")
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.SGD(model.parameters(), lr=0.01)

#lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 50, 150], last_epoch=-1)
#lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1)

params = list(model.parameters())
cnt = 0
for i in range(len(params)):
    cnt += params[i].data.reshape(-1,).shape[0]
print("How many total parameters       | %d" % cnt)
print("="*100)
##########################################################################################################################
losses = torch.zeros((300))
save_again = True

iters = len(train_loader)
for epoch in range(300):
    #adjust_learning_rate(optimizer, epoch)
    for i, (x, y) in enumerate(train_loader):
        #lr_scheduler.step(epoch + i / iters)
        
        optimizer.zero_grad()
        
        output = model(x.float().to(device))
        
        loss = criterion(output, y.long().to(device))
        
        loss.backward()
        
        optimizer.step()
        
        losses[epoch] += loss.item()
    
    #lr_scheduler.step()
    
    losses[epoch] /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), losses[epoch].item()), end=" ")
    
    
    accuracy = 0
    with torch.no_grad():
        model.eval()
        correct = 0
        for x, y in test_loader:
            output = model(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
        accuracy = correct / len(test_loader.dataset)
        '''
        if accuracy >= 0.9303:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses[epoch].item()}, "/home/cscoi/MH/resnet56_real4.pth")
            save_again = False
            print("[Accuracy:%f]" % accuracy)
            print("Saved early")
            break
        ''' 
        print("[Accuracy:%f]" % accuracy)
        model.train()

if save_again:
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses[epoch].item()}, "/home/cscoi/MH/vgg19.pth")
    print("Saved")
    
