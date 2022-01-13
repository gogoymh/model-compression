import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse

#### Custom Library #################################################################################################
from functions import load, cal_sparsity, minmax_sparsity, minmax_recip_loss, make_prob, cross_over, Mutate
from network import LeNet as Net

#### Hyper parameter ################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
parser.add_argument("--mask", type=float, default=0.5, help="Masking rate for sparsity tensor")
parser.add_argument("--mutate", type=float, default=0.01, help="Mutate probability")
parser.add_argument("--alpha", type=float, default=0.5, help="Ratio between loss and sparsity")

opt = parser.parse_args()

#### Load data ######################################################################################################
train_x, train_y, test_x, test_y = load()

train_x = train_x.reshape(train_x.shape[0],1,28,28)/255
test_x = test_x.reshape(test_x.shape[0],1,28,28)/255

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
#test_y = torch.from_numpy(test_y)

train = TensorDataset(train_x, train_y)
train_loader = DataLoader(train, batch_size = opt.batch_size, shuffle = True)

print(len(train_loader)) # train sample 나누기 batch size

#### Loss function ##################################################################################################
criterion = nn.CrossEntropyLoss()

#### Models ####
torch.cuda.manual_seed(1)
model_01 = Net().cuda()
optimizer_01 = optim.SGD(model_01.parameters(), lr=opt.lr)
params_01 = list(model_01.parameters())

torch.cuda.manual_seed(2)
model_02 = Net().cuda()
optimizer_02 = optim.SGD(model_02.parameters(), lr=opt.lr)
params_02 = list(model_02.parameters())

torch.cuda.manual_seed(3)
model_03 = Net().cuda()
optimizer_03 = optim.SGD(model_03.parameters(), lr=opt.lr)
params_03 = list(model_03.parameters())

torch.cuda.manual_seed(4)
model_04 = Net().cuda()
optimizer_04 = optim.SGD(model_04.parameters(), lr=opt.lr)
params_04 = list(model_04.parameters())

#### Count Parameters ###############################################################################################
cnt = 0
for i in range(10):
   cnt += params_01[i].data.reshape(-1,).shape[0]   
print("How many total parameters: %d" % cnt)

#### initial mask ###################################################################################################
mask_01 = []
for i in range(10):
    torch.manual_seed(1)
    mask = torch.FloatTensor(params_01[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_01.append(mask)

mask_02 = []
for i in range(10):
    torch.manual_seed(2)
    mask = torch.FloatTensor(params_02[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_02.append(mask)

mask_03 = []
for i in range(10):
    torch.manual_seed(3)
    mask = torch.FloatTensor(params_03[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_03.append(mask)

mask_04 = []
for i in range(10):
    torch.manual_seed(4)
    mask = torch.FloatTensor(params_04[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_04.append(mask)

#### 실험 ###########################################################################################################
#####################################################################################################################

print("Total epoch is %d" % opt.n_epochs)
alpha = opt.alpha
total_loss_01 = torch.zeros((opt.n_epochs))
total_loss_02 = torch.zeros((opt.n_epochs))
total_loss_03 = torch.zeros((opt.n_epochs))
total_loss_04 = torch.zeros((opt.n_epochs))

total_sparsity_01 = torch.zeros((opt.n_epochs))
total_sparsity_02 = torch.zeros((opt.n_epochs))
total_sparsity_03 = torch.zeros((opt.n_epochs))
total_sparsity_04 = torch.zeros((opt.n_epochs))

import timeit # 시간 측정
start = timeit.default_timer()

for epoch in range(opt.n_epochs):
  
  #### weight removing ####
  for i in range(10):
      params_01[i].data *= mask_01[i]
      params_02[i].data *= mask_02[i]
      params_03[i].data *= mask_03[i]
      params_04[i].data *= mask_04[i]
  
  #### model training ####
  for batch_idx, (x, y) in enumerate(train_loader):
    
    x = Variable(x).float().cuda()
    y = Variable(y).long().cuda() 
    
    optimizer_01.zero_grad()
    optimizer_02.zero_grad()
    optimizer_03.zero_grad()
    optimizer_04.zero_grad()
    
    output_01 = model_01(x)
    output_02 = model_02(x)
    output_03 = model_03(x)
    output_04 = model_04(x)
        
    loss_01 = criterion(output_01, y)        
    loss_02 = criterion(output_02, y)
    loss_03 = criterion(output_03, y)
    loss_04 = criterion(output_04, y)
    
    loss_01.backward()
    loss_02.backward()
    loss_03.backward()
    loss_04.backward()
    
    for i in range(10): # gradient removing
        params_01[i].grad *= mask_01[i]
        params_02[i].grad *= mask_02[i]
        params_03[i].grad *= mask_03[i]
        params_04[i].grad *= mask_04[i]
        
    optimizer_01.step()
    optimizer_02.step()
    optimizer_03.step()
    optimizer_04.step()
    
    total_loss_01[epoch] += loss_01.item()
    total_loss_02[epoch] += loss_02.item()
    total_loss_03[epoch] += loss_03.item()
    total_loss_04[epoch] += loss_04.item()
    
     
  total_loss_01[epoch] /= len(train_loader)
  total_loss_02[epoch] /= len(train_loader)
  total_loss_03[epoch] /= len(train_loader)
  total_loss_04[epoch] /= len(train_loader)
  
  #### masking selection ####
  
  ## get fitness ##
  sparsity_01 = 0
  sparsity_02 = 0
  sparsity_03 = 0
  sparsity_04 = 0
  for i in range(10):
      sparsity_01 += cal_sparsity(mask_01[i])
      sparsity_02 += cal_sparsity(mask_02[i])
      sparsity_03 += cal_sparsity(mask_03[i])
      sparsity_04 += cal_sparsity(mask_04[i])
  
  S = minmax_sparsity(sparsity_01, sparsity_02, sparsity_03, sparsity_04)
  L = minmax_recip_loss(total_loss_01[epoch],total_loss_02[epoch],total_loss_03[epoch],total_loss_04[epoch])
  prob = make_prob(alpha*S,L)
  
  for i in range(10):
      ## selection ##
      copy_01 = torch.empty((mask_01[i].shape))
      copy_02 = torch.empty((mask_02[i].shape))
      copy_03 = torch.empty((mask_03[i].shape))
      copy_04 = torch.empty((mask_04[i].shape))
      
      copy = [copy_01, copy_02, copy_03, copy_04]
      mask_list = [mask_01[i], mask_02[i], mask_03[i], mask_04[i]]
      
      for smp in range(4):
          #copy[smp] = mask_list[torch.multinomial(prob,1)[0]].clone().detach()
          copy[smp] = mask_list[np.random.choice(4, 1, p=prob.numpy())[0]].clone().detach()
          
      ## simple pairs : (copy_01, copy_02), (copy_03, copy_04) ##
      ## cross over ##
      cross_over(copy[0],copy[1])
      cross_over(copy[2],copy[3])
      
      ## mutation ##
      Mutate(copy[0], opt.mutate)
      Mutate(copy[1], opt.mutate)
      Mutate(copy[2], opt.mutate)
      Mutate(copy[3], opt.mutate)
      
      ## inherit ##
      mask_01[i] = copy[0].clone().detach().float().cuda()
      mask_02[i] = copy[1].clone().detach().float().cuda()
      mask_03[i] = copy[2].clone().detach().float().cuda()
      mask_04[i] = copy[3].clone().detach().float().cuda()
  
  #### accuracy ####
  test_x = Variable(test_x).float().cuda()
  
  result_01 = torch.max(model_01(test_x).data, 1)[1]
  result_01 = result_01.cpu()
  accuracy_01 = np.sum(test_y == result_01.numpy()) / test_y.shape[0]
  result_02 = torch.max(model_02(test_x).data, 1)[1]
  result_02 = result_02.cpu()
  accuracy_02 = np.sum(test_y == result_02.numpy()) / test_y.shape[0]
  result_03 = torch.max(model_03(test_x).data, 1)[1]
  result_03 = result_03.cpu()
  accuracy_03 = np.sum(test_y == result_03.numpy()) / test_y.shape[0]
  result_04 = torch.max(model_04(test_x).data, 1)[1]
  result_04 = result_04.cpu()
  accuracy_04 = np.sum(test_y == result_04.numpy()) / test_y.shape[0]
  
  #### record ####
  for i in range(10):
      total_sparsity_01[epoch] += cal_sparsity(params_01[i].data)
      total_sparsity_02[epoch] += cal_sparsity(params_02[i].data)
      total_sparsity_03[epoch] += cal_sparsity(params_03[i].data)
      total_sparsity_04[epoch] += cal_sparsity(params_04[i].data)
  
  #### print ####
  
  print("=" * 100)
  print("Epoch: %d" % (epoch+1))
  print("Loss_01: %f, Loss_02: %f, Loss_03: %f, Loss_04: %f" % (total_loss_01[epoch],total_loss_02[epoch],total_loss_03[epoch],total_loss_04[epoch]))
  print("Sparsity_01: %d, Sparsity_02: %d, Sparsity_03: %d, Sparsity_04: %d" % (total_sparsity_01[epoch],total_sparsity_02[epoch],total_sparsity_03[epoch],total_sparsity_04[epoch]))
  print("Prob_01: %f, Prob_02: %f, Prob_03: %f, Prob_04: %f" % (prob[0], prob[1], prob[2], prob[3]))
  print("Accuracy_01: %f, Accuracy_02: %f, Accuracy_03: %f, Accuracy_04: %f" % (accuracy_01, accuracy_02, accuracy_03, accuracy_04))
  
#### time ##########################################################################################################
stop = timeit.default_timer()
print("=" * 100)
print("걸린 시간은 %f 초" % (stop - start))
print("\n")

#### plot ##########################################################################################################
import matplotlib.pyplot as plt
plt.plot(total_loss_01.numpy())
plt.plot(total_loss_02.numpy())
plt.plot(total_loss_03.numpy())
plt.plot(total_loss_04.numpy())
plt.axis([0,opt.n_epochs,0,2.5])
plt.show()

plt.plot(total_sparsity_01.numpy())
plt.plot(total_sparsity_02.numpy())
plt.plot(total_sparsity_03.numpy())
plt.plot(total_sparsity_04.numpy())
plt.axis([0,opt.n_epochs,0,cnt])
plt.show()

print("\n")
print('Rememver to save best one in 4 models')

'''
torch.save({'epoch': opt.n_epochs,
            'model_state_dict': model_01.state_dict(),
            'optimizer_state_dict': optimizer_01.state_dict(),
            'loss': loss_01}, 'C:/유민형/개인 연구/model compression/models/best_1.3.5.pkl')
'''




















