#import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

train_loader = DataLoader(
                datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
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

model = models.resnet50(num_classes=10)
device = torch.device("cuda:0")
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 100:
        lr = 0.1
    elif epoch >= 100 and epoch < 250:
        lr = 0.01
    else:
        lr = 0.001
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

losses = torch.zeros((500))
save_again = True

for epoch in range(500):
    adjust_learning_rate(optimizer, epoch)
    for x, y in train_loader:
        
        optimizer.zero_grad()
        
        output = model(x.float().to(device))
        
        loss = criterion(output, y.long().to(device))
        
        loss.backward()
        
        optimizer.step()
        
        losses[epoch] += loss.item()
    
    losses[epoch] /= len(train_loader)
    print("[Epoch:%d] Loss is %f" % ((epoch+1), losses[epoch].item()))
    
    if (epoch+1) % 10 == 0:     
        accuracy = 0
        with torch.no_grad():
            model.eval()
            correct = 0
            for x, y in test_loader:
                output = model(x.float().to(device))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
            accuracy = correct / len(test_loader.dataset)
            if accuracy >= 0.936:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': losses[epoch].item()}, "/home/cscoi/MH/resnet50.pth")
                save_again = False
                print("Accuracy is %f" % accuracy)
                print("Saved early")
                break
            
        print("Accuracy is %f" % accuracy)
        model.train()

if save_again:
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses[epoch].item()}, "/home/cscoi/MH/resnet50.pth")
    print("Saved")
    
    
    
    
    
    
    
