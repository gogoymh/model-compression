import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse

#### Custom Library #################################################################################################
from functions import load, cal_sparsity, minmax_sparsity, minmax_recip_loss, make_prob8, cross_over, Mutate
from network import LeNet as Net

#### Hyper parameter ################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
parser.add_argument("--mask", type=float, default=0.1, help="Masking rate for sparsity tensor")
parser.add_argument("--mutate", type=float, default=0.01, help="Mutate probability")
parser.add_argument("--alpha", type=float, default=1, help="Ratio between loss and sparsity")
parser.add_argument("--buffer", type=int, default=50, help="Buffer for converge")
parser.add_argument("--recovery", type=float, default=1, help="Recovery constant")
parser.add_argument("--ensemble", type=int, default=8, help="Number of models")
parser.add_argument("--layer", type=int, default=10, help="Number of layers")

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

torch.cuda.manual_seed(5)
model_05 = Net().cuda()
optimizer_05 = optim.SGD(model_05.parameters(), lr=opt.lr)
params_05 = list(model_05.parameters())

torch.cuda.manual_seed(6)
model_06 = Net().cuda()
optimizer_06 = optim.SGD(model_06.parameters(), lr=opt.lr)
params_06 = list(model_06.parameters())

torch.cuda.manual_seed(7)
model_07 = Net().cuda()
optimizer_07 = optim.SGD(model_07.parameters(), lr=opt.lr)
params_07 = list(model_07.parameters())

torch.cuda.manual_seed(8)
model_08 = Net().cuda()
optimizer_08 = optim.SGD(model_08.parameters(), lr=opt.lr)
params_08 = list(model_08.parameters())

#### Count Parameters ###############################################################################################
cnt = 0
for i in range(opt.layer):
   cnt += params_01[i].data.reshape(-1,).shape[0]   
print("How many total parameters: %d" % cnt)

#### initial mask ###################################################################################################
mask_01 = []
for i in range(opt.layer):
    torch.manual_seed(1)
    mask = torch.FloatTensor(params_01[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_01.append(mask)

mask_02 = []
for i in range(opt.layer):
    torch.manual_seed(2)
    mask = torch.FloatTensor(params_02[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_02.append(mask)

mask_03 = []
for i in range(opt.layer):
    torch.manual_seed(3)
    mask = torch.FloatTensor(params_03[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_03.append(mask)

mask_04 = []
for i in range(opt.layer):
    torch.manual_seed(4)
    mask = torch.FloatTensor(params_04[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_04.append(mask)

mask_05 = []
for i in range(opt.layer):
    torch.manual_seed(5)
    mask = torch.FloatTensor(params_05[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_05.append(mask)

mask_06 = []
for i in range(opt.layer):
    torch.manual_seed(6)
    mask = torch.FloatTensor(params_06[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_06.append(mask)

mask_07 = []
for i in range(opt.layer):
    torch.manual_seed(7)
    mask = torch.FloatTensor(params_07[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_07.append(mask)

mask_08 = []
for i in range(opt.layer):
    torch.manual_seed(8)
    mask = torch.FloatTensor(params_08[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_08.append(mask)

#### 실험 ###########################################################################################################
#####################################################################################################################

print("Total epoch is %d" % (opt.n_epochs))
alpha = opt.alpha
total_loss_01 = torch.zeros((opt.n_epochs))
total_loss_02 = torch.zeros((opt.n_epochs))
total_loss_03 = torch.zeros((opt.n_epochs))
total_loss_04 = torch.zeros((opt.n_epochs))
total_loss_05 = torch.zeros((opt.n_epochs))
total_loss_06 = torch.zeros((opt.n_epochs))
total_loss_07 = torch.zeros((opt.n_epochs))
total_loss_08 = torch.zeros((opt.n_epochs))

total_sparsity_01 = torch.zeros((opt.n_epochs))
total_sparsity_02 = torch.zeros((opt.n_epochs))
total_sparsity_03 = torch.zeros((opt.n_epochs))
total_sparsity_04 = torch.zeros((opt.n_epochs))
total_sparsity_05 = torch.zeros((opt.n_epochs))
total_sparsity_06 = torch.zeros((opt.n_epochs))
total_sparsity_07 = torch.zeros((opt.n_epochs))
total_sparsity_08 = torch.zeros((opt.n_epochs))

import timeit # 시간 측정
start = timeit.default_timer()

for epoch in range(opt.n_epochs):
    
  if epoch >= opt.buffer:
      #### weight removing ####
      for i in range(opt.layer):
          params_01[i].data += opt.recovery * ((params_01[i].data == 0).float() == mask_01[i]).float()
          params_02[i].data += opt.recovery * ((params_02[i].data == 0).float() == mask_02[i]).float()
          params_03[i].data += opt.recovery * ((params_03[i].data == 0).float() == mask_03[i]).float()
          params_04[i].data += opt.recovery * ((params_04[i].data == 0).float() == mask_04[i]).float()
          params_05[i].data += opt.recovery * ((params_05[i].data == 0).float() == mask_05[i]).float()
          params_06[i].data += opt.recovery * ((params_06[i].data == 0).float() == mask_06[i]).float()
          params_07[i].data += opt.recovery * ((params_07[i].data == 0).float() == mask_07[i]).float()
          params_08[i].data += opt.recovery * ((params_08[i].data == 0).float() == mask_08[i]).float()  
          
          params_01[i].data *= mask_01[i]
          params_02[i].data *= mask_02[i]
          params_03[i].data *= mask_03[i]
          params_04[i].data *= mask_04[i]
          params_05[i].data *= mask_05[i]
          params_06[i].data *= mask_06[i]
          params_07[i].data *= mask_07[i]
          params_08[i].data *= mask_08[i]
  
  #### model training ####
  for batch_idx, (x, y) in enumerate(train_loader):
    
    x = Variable(x).float().cuda()
    y = Variable(y).long().cuda() 
    
    optimizer_01.zero_grad()
    optimizer_02.zero_grad()
    optimizer_03.zero_grad()
    optimizer_04.zero_grad()
    optimizer_05.zero_grad()
    optimizer_06.zero_grad()
    optimizer_07.zero_grad()
    optimizer_08.zero_grad()
    
    output_01 = model_01(x)
    output_02 = model_02(x)
    output_03 = model_03(x)
    output_04 = model_04(x)
    output_05 = model_05(x)
    output_06 = model_06(x)
    output_07 = model_07(x)
    output_08 = model_08(x)
        
    loss_01 = criterion(output_01, y)        
    loss_02 = criterion(output_02, y)
    loss_03 = criterion(output_03, y)
    loss_04 = criterion(output_04, y)
    loss_05 = criterion(output_05, y)        
    loss_06 = criterion(output_06, y)
    loss_07 = criterion(output_07, y)
    loss_08 = criterion(output_08, y)
    
    loss_01.backward()
    loss_02.backward()
    loss_03.backward()
    loss_04.backward()
    loss_05.backward()
    loss_06.backward()
    loss_07.backward()
    loss_08.backward()
    
    if epoch >= opt.buffer:
        for i in range(opt.layer): # gradient removing
            params_01[i].grad *= mask_01[i]
            params_02[i].grad *= mask_02[i]
            params_03[i].grad *= mask_03[i]
            params_04[i].grad *= mask_04[i]
            params_05[i].grad *= mask_05[i]
            params_06[i].grad *= mask_06[i]
            params_07[i].grad *= mask_07[i]
            params_08[i].grad *= mask_08[i]
        
    optimizer_01.step()
    optimizer_02.step()
    optimizer_03.step()
    optimizer_04.step()
    optimizer_05.step()
    optimizer_06.step()
    optimizer_07.step()
    optimizer_08.step()

    total_loss_01[epoch] += loss_01.item()
    total_loss_02[epoch] += loss_02.item()
    total_loss_03[epoch] += loss_03.item()
    total_loss_04[epoch] += loss_04.item()
    total_loss_05[epoch] += loss_05.item()
    total_loss_06[epoch] += loss_06.item()
    total_loss_07[epoch] += loss_07.item()
    total_loss_08[epoch] += loss_08.item()
    
  total_loss_01[epoch] /= len(train_loader)
  total_loss_02[epoch] /= len(train_loader)
  total_loss_03[epoch] /= len(train_loader)
  total_loss_04[epoch] /= len(train_loader)
  total_loss_05[epoch] /= len(train_loader)
  total_loss_06[epoch] /= len(train_loader)
  total_loss_07[epoch] /= len(train_loader)
  total_loss_08[epoch] /= len(train_loader)
  
  if epoch >= opt.buffer:
      #### masking selection ####
      ## get fitness ##
      sparsity_01 = 0
      sparsity_02 = 0
      sparsity_03 = 0
      sparsity_04 = 0
      sparsity_05 = 0
      sparsity_06 = 0
      sparsity_07 = 0
      sparsity_08 = 0
      
      for i in range(opt.layer):
          sparsity_01 += cal_sparsity(mask_01[i])
          sparsity_02 += cal_sparsity(mask_02[i])
          sparsity_03 += cal_sparsity(mask_03[i])
          sparsity_04 += cal_sparsity(mask_04[i])
          sparsity_05 += cal_sparsity(mask_05[i])
          sparsity_06 += cal_sparsity(mask_06[i])
          sparsity_07 += cal_sparsity(mask_07[i])
          sparsity_08 += cal_sparsity(mask_08[i])
          
      S = minmax_sparsity(sparsity_01, sparsity_02, sparsity_03, sparsity_04, sparsity_05, sparsity_06, sparsity_07, sparsity_08)
      L = minmax_recip_loss(total_loss_01[epoch],total_loss_02[epoch],total_loss_03[epoch],total_loss_04[epoch], total_loss_05[epoch],total_loss_06[epoch],total_loss_07[epoch],total_loss_08[epoch])
      prob = make_prob8(alpha*S,L)
      
      for i in range(opt.layer):
          ## selection ##
          copy_01 = torch.empty((mask_01[i].shape))
          copy_02 = torch.empty((mask_02[i].shape))
          copy_03 = torch.empty((mask_03[i].shape))
          copy_04 = torch.empty((mask_04[i].shape))
          copy_05 = torch.empty((mask_05[i].shape))
          copy_06 = torch.empty((mask_06[i].shape))
          copy_07 = torch.empty((mask_07[i].shape))
          copy_08 = torch.empty((mask_08[i].shape))
      
          copy = [copy_01, copy_02, copy_03, copy_04, copy_05, copy_06, copy_07, copy_08]
          mask_list = [mask_01[i], mask_02[i], mask_03[i], mask_04[i], mask_05[i], mask_06[i], mask_07[i], mask_08[i]]
      
          for smp in range(opt.ensemble):
              #copy[smp] = mask_list[torch.multinomial(prob,1)[0]].clone().detach()
              copy[smp] = mask_list[np.random.choice(opt.ensemble, 1, p=prob.numpy())[0]].clone().detach()
          
          ## simple pairs : (copy_01, copy_02), (copy_03, copy_04) ##
          ## cross over ##
          cross_over(copy[0],copy[1])
          cross_over(copy[2],copy[3])
          cross_over(copy[4],copy[5])
          cross_over(copy[6],copy[7])
      
          ## mutation ##
          Mutate(copy[0], opt.mutate)
          Mutate(copy[1], opt.mutate)
          Mutate(copy[2], opt.mutate)
          Mutate(copy[3], opt.mutate)
          Mutate(copy[4], opt.mutate)
          Mutate(copy[5], opt.mutate)
          Mutate(copy[6], opt.mutate)
          Mutate(copy[7], opt.mutate)
          
          ## inherit ##
          mask_01[i] = copy[0].clone().detach().float().cuda()
          mask_02[i] = copy[1].clone().detach().float().cuda()
          mask_03[i] = copy[2].clone().detach().float().cuda()
          mask_04[i] = copy[3].clone().detach().float().cuda()
          mask_05[i] = copy[4].clone().detach().float().cuda()
          mask_06[i] = copy[5].clone().detach().float().cuda()
          mask_07[i] = copy[6].clone().detach().float().cuda()
          mask_08[i] = copy[7].clone().detach().float().cuda()
  
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
  result_05 = torch.max(model_05(test_x).data, 1)[1]
  result_05 = result_05.cpu()
  accuracy_05 = np.sum(test_y == result_05.numpy()) / test_y.shape[0]
  result_06 = torch.max(model_06(test_x).data, 1)[1]
  result_06 = result_06.cpu()
  accuracy_06 = np.sum(test_y == result_06.numpy()) / test_y.shape[0]
  result_07 = torch.max(model_07(test_x).data, 1)[1]
  result_07 = result_07.cpu()
  accuracy_07 = np.sum(test_y == result_07.numpy()) / test_y.shape[0]
  result_08 = torch.max(model_08(test_x).data, 1)[1]
  result_08 = result_08.cpu()
  accuracy_08 = np.sum(test_y == result_08.numpy()) / test_y.shape[0]
  
  #### record ####
  for i in range(opt.layer):
      total_sparsity_01[epoch] += cal_sparsity(params_01[i].data)
      total_sparsity_02[epoch] += cal_sparsity(params_02[i].data)
      total_sparsity_03[epoch] += cal_sparsity(params_03[i].data)
      total_sparsity_04[epoch] += cal_sparsity(params_04[i].data)
      total_sparsity_05[epoch] += cal_sparsity(params_05[i].data)
      total_sparsity_06[epoch] += cal_sparsity(params_06[i].data)
      total_sparsity_07[epoch] += cal_sparsity(params_07[i].data)
      total_sparsity_08[epoch] += cal_sparsity(params_08[i].data)
  
  #### print ####
  
  print("=" * 100)
  print("Epoch: %d" % (epoch+1))
  
  print("Loss_01: %f, Loss_02: %f, Loss_03: %f, Loss_04: %f" % (total_loss_01[epoch],total_loss_02[epoch],total_loss_03[epoch],total_loss_04[epoch]))
  print("Loss_05: %f, Loss_06: %f, Loss_07: %f, Loss_08: %f" % (total_loss_05[epoch],total_loss_06[epoch],total_loss_07[epoch],total_loss_08[epoch]))
  
  print("Sparsity_01: %d, Sparsity_02: %d, Sparsity_03: %d, Sparsity_04: %d" % (total_sparsity_01[epoch],total_sparsity_02[epoch],total_sparsity_03[epoch],total_sparsity_04[epoch]))
  print("Sparsity_05: %d, Sparsity_06: %d, Sparsity_07: %d, Sparsity_08: %d" % (total_sparsity_05[epoch],total_sparsity_06[epoch],total_sparsity_07[epoch],total_sparsity_08[epoch]))
  
  if epoch >= opt.buffer:
      print("Prob_01: %f, Prob_02: %f, Prob_03: %f, Prob_04: %f" % (prob[0], prob[1], prob[2], prob[3]))
      print("Prob_05: %f, Prob_06: %f, Prob_07: %f, Prob_08: %f" % (prob[4], prob[5], prob[6], prob[7]))
      
  print("Accuracy_01: %f, Accuracy_02: %f, Accuracy_03: %f, Accuracy_04: %f" % (accuracy_01, accuracy_02, accuracy_03, accuracy_04))
  print("Accuracy_05: %f, Accuracy_06: %f, Accuracy_07: %f, Accuracy_08: %f" % (accuracy_05, accuracy_06, accuracy_07, accuracy_08))
  
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
plt.plot(total_loss_05.numpy())
plt.plot(total_loss_06.numpy())
plt.plot(total_loss_07.numpy())
plt.plot(total_loss_08.numpy())
plt.axis([0,opt.n_epochs,0,2.5])
plt.show()

plt.plot(total_sparsity_01.numpy())
plt.plot(total_sparsity_02.numpy())
plt.plot(total_sparsity_03.numpy())
plt.plot(total_sparsity_04.numpy())
plt.plot(total_sparsity_05.numpy())
plt.plot(total_sparsity_06.numpy())
plt.plot(total_sparsity_07.numpy())
plt.plot(total_sparsity_08.numpy())
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




















