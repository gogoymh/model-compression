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
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
parser.add_argument("--mask", type=float, default=0.1, help="Masking rate for sparsity tensor")
parser.add_argument("--mutate", type=float, default=0.1, help="Mutate probability")
parser.add_argument("--alpha", type=float, default=1, help="Ratio between loss and sparsity")
parser.add_argument("--buffer", type=int, default=50, help="Buffer for converge")
parser.add_argument("--recovery", type=float, default=0.0001, help="Recovery constant")
parser.add_argument("--ensemble", type=int, default=16, help="Number of models")
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

torch.cuda.manual_seed(9)
model_09 = Net().cuda()
optimizer_09 = optim.SGD(model_09.parameters(), lr=opt.lr)
params_09 = list(model_09.parameters())

torch.cuda.manual_seed(10)
model_10 = Net().cuda()
optimizer_10 = optim.SGD(model_10.parameters(), lr=opt.lr)
params_10 = list(model_10.parameters())

torch.cuda.manual_seed(11)
model_11 = Net().cuda()
optimizer_11 = optim.SGD(model_11.parameters(), lr=opt.lr)
params_11 = list(model_11.parameters())

torch.cuda.manual_seed(12)
model_12 = Net().cuda()
optimizer_12 = optim.SGD(model_12.parameters(), lr=opt.lr)
params_12 = list(model_12.parameters())

torch.cuda.manual_seed(13)
model_13 = Net().cuda()
optimizer_13 = optim.SGD(model_13.parameters(), lr=opt.lr)
params_13 = list(model_13.parameters())

torch.cuda.manual_seed(14)
model_14 = Net().cuda()
optimizer_14 = optim.SGD(model_14.parameters(), lr=opt.lr)
params_14 = list(model_14.parameters())

torch.cuda.manual_seed(15)
model_15 = Net().cuda()
optimizer_15 = optim.SGD(model_15.parameters(), lr=opt.lr)
params_15 = list(model_15.parameters())

torch.cuda.manual_seed(16)
model_16 = Net().cuda()
optimizer_16 = optim.SGD(model_16.parameters(), lr=opt.lr)
params_16 = list(model_16.parameters())

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
    
mask_09 = []
for i in range(opt.layer):
    torch.manual_seed(9)
    mask = torch.FloatTensor(params_09[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_09.append(mask)

mask_10 = []
for i in range(opt.layer):
    torch.manual_seed(10)
    mask = torch.FloatTensor(params_10[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_10.append(mask)

mask_11 = []
for i in range(opt.layer):
    torch.manual_seed(11)
    mask = torch.FloatTensor(params_11[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_11.append(mask)

mask_12 = []
for i in range(opt.layer):
    torch.manual_seed(12)
    mask = torch.FloatTensor(params_12[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_12.append(mask)

mask_13 = []
for i in range(opt.layer):
    torch.manual_seed(13)
    mask = torch.FloatTensor(params_13[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_13.append(mask)

mask_14 = []
for i in range(opt.layer):
    torch.manual_seed(14)
    mask = torch.FloatTensor(params_14[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_14.append(mask)

mask_15 = []
for i in range(opt.layer):
    torch.manual_seed(15)
    mask = torch.FloatTensor(params_15[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_15.append(mask)

mask_16 = []
for i in range(opt.layer):
    torch.manual_seed(16)
    mask = torch.FloatTensor(params_16[i].data.shape).uniform_() > opt.mask
    mask = mask.float().cuda()
    mask_16.append(mask)

#### 실험 ###########################################################################################################
#####################################################################################################################

print("Total epoch is %d" % (opt.n_epochs))

total_loss_01 = torch.zeros((opt.n_epochs))
total_loss_02 = torch.zeros((opt.n_epochs))
total_loss_03 = torch.zeros((opt.n_epochs))
total_loss_04 = torch.zeros((opt.n_epochs))
total_loss_05 = torch.zeros((opt.n_epochs))
total_loss_06 = torch.zeros((opt.n_epochs))
total_loss_07 = torch.zeros((opt.n_epochs))
total_loss_08 = torch.zeros((opt.n_epochs))
total_loss_09 = torch.zeros((opt.n_epochs))
total_loss_10 = torch.zeros((opt.n_epochs))
total_loss_11 = torch.zeros((opt.n_epochs))
total_loss_12 = torch.zeros((opt.n_epochs))
total_loss_13 = torch.zeros((opt.n_epochs))
total_loss_14 = torch.zeros((opt.n_epochs))
total_loss_15 = torch.zeros((opt.n_epochs))
total_loss_16 = torch.zeros((opt.n_epochs))

total_sparsity_01 = torch.zeros((opt.n_epochs))
total_sparsity_02 = torch.zeros((opt.n_epochs))
total_sparsity_03 = torch.zeros((opt.n_epochs))
total_sparsity_04 = torch.zeros((opt.n_epochs))
total_sparsity_05 = torch.zeros((opt.n_epochs))
total_sparsity_06 = torch.zeros((opt.n_epochs))
total_sparsity_07 = torch.zeros((opt.n_epochs))
total_sparsity_08 = torch.zeros((opt.n_epochs))
total_sparsity_09 = torch.zeros((opt.n_epochs))
total_sparsity_10 = torch.zeros((opt.n_epochs))
total_sparsity_11 = torch.zeros((opt.n_epochs))
total_sparsity_12 = torch.zeros((opt.n_epochs))
total_sparsity_13 = torch.zeros((opt.n_epochs))
total_sparsity_14 = torch.zeros((opt.n_epochs))
total_sparsity_15 = torch.zeros((opt.n_epochs))
total_sparsity_16 = torch.zeros((opt.n_epochs))

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
          params_09[i].data += opt.recovery * ((params_09[i].data == 0).float() == mask_09[i]).float()
          params_10[i].data += opt.recovery * ((params_10[i].data == 0).float() == mask_10[i]).float()
          params_11[i].data += opt.recovery * ((params_11[i].data == 0).float() == mask_11[i]).float()
          params_12[i].data += opt.recovery * ((params_12[i].data == 0).float() == mask_12[i]).float()
          params_13[i].data += opt.recovery * ((params_13[i].data == 0).float() == mask_13[i]).float()
          params_14[i].data += opt.recovery * ((params_14[i].data == 0).float() == mask_14[i]).float()
          params_15[i].data += opt.recovery * ((params_15[i].data == 0).float() == mask_15[i]).float()
          params_16[i].data += opt.recovery * ((params_16[i].data == 0).float() == mask_16[i]).float()  
          
          params_01[i].data *= mask_01[i]
          params_02[i].data *= mask_02[i]
          params_03[i].data *= mask_03[i]
          params_04[i].data *= mask_04[i]
          params_05[i].data *= mask_05[i]
          params_06[i].data *= mask_06[i]
          params_07[i].data *= mask_07[i]
          params_08[i].data *= mask_08[i]
          params_09[i].data *= mask_09[i]
          params_10[i].data *= mask_10[i]
          params_11[i].data *= mask_11[i]
          params_12[i].data *= mask_12[i]
          params_13[i].data *= mask_13[i]
          params_14[i].data *= mask_14[i]
          params_15[i].data *= mask_15[i]
          params_16[i].data *= mask_16[i]
  
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
    optimizer_09.zero_grad()
    optimizer_10.zero_grad()
    optimizer_11.zero_grad()
    optimizer_12.zero_grad()
    optimizer_13.zero_grad()
    optimizer_14.zero_grad()
    optimizer_15.zero_grad()
    optimizer_16.zero_grad()
    
    output_01 = model_01(x)
    output_02 = model_02(x)
    output_03 = model_03(x)
    output_04 = model_04(x)
    output_05 = model_05(x)
    output_06 = model_06(x)
    output_07 = model_07(x)
    output_08 = model_08(x)
    output_09 = model_09(x)
    output_10 = model_10(x)
    output_11 = model_11(x)
    output_12 = model_12(x)
    output_13 = model_13(x)
    output_14 = model_14(x)
    output_15 = model_15(x)
    output_16 = model_16(x)
        
    loss_01 = criterion(output_01, y)        
    loss_02 = criterion(output_02, y)
    loss_03 = criterion(output_03, y)
    loss_04 = criterion(output_04, y)
    loss_05 = criterion(output_05, y)        
    loss_06 = criterion(output_06, y)
    loss_07 = criterion(output_07, y)
    loss_08 = criterion(output_08, y)
    loss_09 = criterion(output_09, y)        
    loss_10 = criterion(output_10, y)
    loss_11 = criterion(output_11, y)
    loss_12 = criterion(output_12, y)
    loss_13 = criterion(output_13, y)        
    loss_14 = criterion(output_14, y)
    loss_15 = criterion(output_15, y)
    loss_16 = criterion(output_16, y)
    
    loss_01.backward()
    loss_02.backward()
    loss_03.backward()
    loss_04.backward()
    loss_05.backward()
    loss_06.backward()
    loss_07.backward()
    loss_08.backward()
    loss_09.backward()
    loss_10.backward()
    loss_11.backward()
    loss_12.backward()
    loss_13.backward()
    loss_14.backward()
    loss_15.backward()
    loss_16.backward()
    
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
            params_09[i].grad *= mask_09[i]
            params_10[i].grad *= mask_10[i]
            params_11[i].grad *= mask_11[i]
            params_12[i].grad *= mask_12[i]
            params_13[i].grad *= mask_13[i]
            params_14[i].grad *= mask_14[i]
            params_15[i].grad *= mask_15[i]
            params_16[i].grad *= mask_16[i]
        
    optimizer_01.step()
    optimizer_02.step()
    optimizer_03.step()
    optimizer_04.step()
    optimizer_05.step()
    optimizer_06.step()
    optimizer_07.step()
    optimizer_08.step()
    optimizer_09.step()
    optimizer_10.step()
    optimizer_11.step()
    optimizer_12.step()
    optimizer_13.step()
    optimizer_14.step()
    optimizer_15.step()
    optimizer_16.step()

    total_loss_01[epoch] += loss_01.item()
    total_loss_02[epoch] += loss_02.item()
    total_loss_03[epoch] += loss_03.item()
    total_loss_04[epoch] += loss_04.item()
    total_loss_05[epoch] += loss_05.item()
    total_loss_06[epoch] += loss_06.item()
    total_loss_07[epoch] += loss_07.item()
    total_loss_08[epoch] += loss_08.item()
    total_loss_09[epoch] += loss_09.item()
    total_loss_10[epoch] += loss_10.item()
    total_loss_11[epoch] += loss_11.item()
    total_loss_12[epoch] += loss_12.item()
    total_loss_13[epoch] += loss_13.item()
    total_loss_14[epoch] += loss_14.item()
    total_loss_15[epoch] += loss_15.item()
    total_loss_16[epoch] += loss_16.item()
    
  total_loss_01[epoch] /= len(train_loader)
  total_loss_02[epoch] /= len(train_loader)
  total_loss_03[epoch] /= len(train_loader)
  total_loss_04[epoch] /= len(train_loader)
  total_loss_05[epoch] /= len(train_loader)
  total_loss_06[epoch] /= len(train_loader)
  total_loss_07[epoch] /= len(train_loader)
  total_loss_08[epoch] /= len(train_loader)
  total_loss_09[epoch] /= len(train_loader)
  total_loss_10[epoch] /= len(train_loader)
  total_loss_11[epoch] /= len(train_loader)
  total_loss_12[epoch] /= len(train_loader)
  total_loss_13[epoch] /= len(train_loader)
  total_loss_14[epoch] /= len(train_loader)
  total_loss_15[epoch] /= len(train_loader)
  total_loss_16[epoch] /= len(train_loader)
  
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
      sparsity_09 = 0
      sparsity_10 = 0
      sparsity_11 = 0
      sparsity_12 = 0
      sparsity_13 = 0
      sparsity_14 = 0
      sparsity_15 = 0
      sparsity_16 = 0
      
      for i in range(opt.layer):
          sparsity_01 += cal_sparsity(mask_01[i])
          sparsity_02 += cal_sparsity(mask_02[i])
          sparsity_03 += cal_sparsity(mask_03[i])
          sparsity_04 += cal_sparsity(mask_04[i])
          sparsity_05 += cal_sparsity(mask_05[i])
          sparsity_06 += cal_sparsity(mask_06[i])
          sparsity_07 += cal_sparsity(mask_07[i])
          sparsity_08 += cal_sparsity(mask_08[i])
          sparsity_09 += cal_sparsity(mask_09[i])
          sparsity_10 += cal_sparsity(mask_10[i])
          sparsity_11 += cal_sparsity(mask_11[i])
          sparsity_12 += cal_sparsity(mask_12[i])
          sparsity_13 += cal_sparsity(mask_13[i])
          sparsity_14 += cal_sparsity(mask_14[i])
          sparsity_15 += cal_sparsity(mask_15[i])
          sparsity_16 += cal_sparsity(mask_16[i])
          
      S = minmax_sparsity(sparsity_01, sparsity_02, sparsity_03, sparsity_04, sparsity_05, sparsity_06, sparsity_07, sparsity_08,
                          sparsity_09, sparsity_10, sparsity_11, sparsity_12, sparsity_13, sparsity_14, sparsity_15, sparsity_16)
      L = minmax_recip_loss(total_loss_01[epoch],total_loss_02[epoch],total_loss_03[epoch],total_loss_04[epoch],
                            total_loss_05[epoch],total_loss_06[epoch],total_loss_07[epoch],total_loss_08[epoch],
                            total_loss_09[epoch],total_loss_10[epoch],total_loss_11[epoch],total_loss_12[epoch],
                            total_loss_13[epoch],total_loss_14[epoch],total_loss_15[epoch],total_loss_16[epoch])
      prob = make_prob(opt.alpha*S,L,opt.ensemble)
      
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
          copy_09 = torch.empty((mask_09[i].shape))
          copy_10 = torch.empty((mask_10[i].shape))
          copy_11 = torch.empty((mask_11[i].shape))
          copy_12 = torch.empty((mask_12[i].shape))
          copy_13 = torch.empty((mask_13[i].shape))
          copy_14 = torch.empty((mask_14[i].shape))
          copy_15 = torch.empty((mask_15[i].shape))
          copy_16 = torch.empty((mask_16[i].shape))
      
          copy = [copy_01, copy_02, copy_03, copy_04, copy_05, copy_06, copy_07, copy_08,
                  copy_09, copy_10, copy_11, copy_12, copy_13, copy_14, copy_15, copy_16]
          mask_list = [mask_01[i], mask_02[i], mask_03[i], mask_04[i], mask_05[i], mask_06[i], mask_07[i], mask_08[i],
                       mask_09[i], mask_10[i], mask_11[i], mask_12[i], mask_13[i], mask_14[i], mask_15[i], mask_16[i]]
      
          for smp in range(opt.ensemble):
              #copy[smp] = mask_list[torch.multinomial(prob,1)[0]].clone().detach()
              copy[smp] = mask_list[np.random.choice(opt.ensemble, 1, p=prob.numpy())[0]].clone().detach()
          
          ## simple pairs : (copy_01, copy_02), (copy_03, copy_04) ##
          ## cross over ##
          cross_over(copy[0],copy[1])
          cross_over(copy[2],copy[3])
          cross_over(copy[4],copy[5])
          cross_over(copy[6],copy[7])
          cross_over(copy[8],copy[9])
          cross_over(copy[10],copy[11])
          cross_over(copy[12],copy[13])
          cross_over(copy[14],copy[15])
      
          ## mutation ##
          Mutate(copy[0], opt.mutate)
          Mutate(copy[1], opt.mutate)
          Mutate(copy[2], opt.mutate)
          Mutate(copy[3], opt.mutate)
          Mutate(copy[4], opt.mutate)
          Mutate(copy[5], opt.mutate)
          Mutate(copy[6], opt.mutate)
          Mutate(copy[7], opt.mutate)
          Mutate(copy[8], opt.mutate)
          Mutate(copy[9], opt.mutate)
          Mutate(copy[10], opt.mutate)
          Mutate(copy[11], opt.mutate)
          Mutate(copy[12], opt.mutate)
          Mutate(copy[13], opt.mutate)
          Mutate(copy[14], opt.mutate)
          Mutate(copy[15], opt.mutate)
          
          ## inherit ##
          mask_01[i] = copy[0].clone().detach().float().cuda()
          mask_02[i] = copy[1].clone().detach().float().cuda()
          mask_03[i] = copy[2].clone().detach().float().cuda()
          mask_04[i] = copy[3].clone().detach().float().cuda()
          mask_05[i] = copy[4].clone().detach().float().cuda()
          mask_06[i] = copy[5].clone().detach().float().cuda()
          mask_07[i] = copy[6].clone().detach().float().cuda()
          mask_08[i] = copy[7].clone().detach().float().cuda()
          mask_09[i] = copy[8].clone().detach().float().cuda()
          mask_10[i] = copy[9].clone().detach().float().cuda()
          mask_11[i] = copy[10].clone().detach().float().cuda()
          mask_12[i] = copy[11].clone().detach().float().cuda()
          mask_13[i] = copy[12].clone().detach().float().cuda()
          mask_14[i] = copy[13].clone().detach().float().cuda()
          mask_15[i] = copy[14].clone().detach().float().cuda()
          mask_16[i] = copy[15].clone().detach().float().cuda()
  
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
  result_09 = torch.max(model_09(test_x).data, 1)[1]
  result_09 = result_09.cpu()
  accuracy_09 = np.sum(test_y == result_09.numpy()) / test_y.shape[0]
  result_10 = torch.max(model_10(test_x).data, 1)[1]
  result_10 = result_10.cpu()
  accuracy_10 = np.sum(test_y == result_10.numpy()) / test_y.shape[0]
  result_11 = torch.max(model_11(test_x).data, 1)[1]
  result_11 = result_11.cpu()
  accuracy_11 = np.sum(test_y == result_11.numpy()) / test_y.shape[0]
  result_12 = torch.max(model_12(test_x).data, 1)[1]
  result_12 = result_12.cpu()
  accuracy_12 = np.sum(test_y == result_12.numpy()) / test_y.shape[0]
  result_13 = torch.max(model_13(test_x).data, 1)[1]
  result_13 = result_13.cpu()
  accuracy_13 = np.sum(test_y == result_13.numpy()) / test_y.shape[0]
  result_14 = torch.max(model_14(test_x).data, 1)[1]
  result_14 = result_14.cpu()
  accuracy_14 = np.sum(test_y == result_14.numpy()) / test_y.shape[0]
  result_15= torch.max(model_15(test_x).data, 1)[1]
  result_15 = result_15.cpu()
  accuracy_15 = np.sum(test_y == result_15.numpy()) / test_y.shape[0]
  result_16 = torch.max(model_16(test_x).data, 1)[1]
  result_16 = result_16.cpu()
  accuracy_16 = np.sum(test_y == result_16.numpy()) / test_y.shape[0]
  
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
      total_sparsity_09[epoch] += cal_sparsity(params_09[i].data)
      total_sparsity_10[epoch] += cal_sparsity(params_10[i].data)
      total_sparsity_11[epoch] += cal_sparsity(params_11[i].data)
      total_sparsity_12[epoch] += cal_sparsity(params_12[i].data)
      total_sparsity_13[epoch] += cal_sparsity(params_13[i].data)
      total_sparsity_14[epoch] += cal_sparsity(params_14[i].data)
      total_sparsity_15[epoch] += cal_sparsity(params_15[i].data)
      total_sparsity_16[epoch] += cal_sparsity(params_16[i].data)
  
  #### print ####
  
  print("=" * 100)
  print("Epoch: %d" % (epoch+1))
  
  print("Loss_01: %f, Loss_02: %f, Loss_03: %f, Loss_04: %f" % (total_loss_01[epoch],total_loss_02[epoch],total_loss_03[epoch],total_loss_04[epoch]))
  print("Loss_05: %f, Loss_06: %f, Loss_07: %f, Loss_08: %f" % (total_loss_05[epoch],total_loss_06[epoch],total_loss_07[epoch],total_loss_08[epoch]))
  print("Loss_09: %f, Loss_10: %f, Loss_11: %f, Loss_12: %f" % (total_loss_09[epoch],total_loss_10[epoch],total_loss_11[epoch],total_loss_12[epoch]))
  print("Loss_13: %f, Loss_14: %f, Loss_15: %f, Loss_16: %f" % (total_loss_13[epoch],total_loss_14[epoch],total_loss_15[epoch],total_loss_16[epoch]))
  
  if epoch >= opt.buffer:
      print("Sparsity_01: %d, Sparsity_02: %d, Sparsity_03: %d, Sparsity_04: %d" % (total_sparsity_01[epoch],total_sparsity_02[epoch],total_sparsity_03[epoch],total_sparsity_04[epoch]))
      print("Sparsity_05: %d, Sparsity_06: %d, Sparsity_07: %d, Sparsity_08: %d" % (total_sparsity_05[epoch],total_sparsity_06[epoch],total_sparsity_07[epoch],total_sparsity_08[epoch]))
      print("Sparsity_09: %d, Sparsity_10: %d, Sparsity_11: %d, Sparsity_12: %d" % (total_sparsity_09[epoch],total_sparsity_10[epoch],total_sparsity_11[epoch],total_sparsity_12[epoch]))
      print("Sparsity_13: %d, Sparsity_14: %d, Sparsity_15: %d, Sparsity_16: %d" % (total_sparsity_13[epoch],total_sparsity_14[epoch],total_sparsity_15[epoch],total_sparsity_16[epoch]))
      
      print("Prob_01: %f, Prob_02: %f, Prob_03: %f, Prob_04: %f" % (prob[0], prob[1], prob[2], prob[3]))
      print("Prob_05: %f, Prob_06: %f, Prob_07: %f, Prob_08: %f" % (prob[4], prob[5], prob[6], prob[7]))
      print("Prob_09: %f, Prob_10: %f, Prob_11: %f, Prob_12: %f" % (prob[8], prob[9], prob[10], prob[11]))
      print("Prob_13: %f, Prob_14: %f, Prob_15: %f, Prob_16: %f" % (prob[12], prob[13], prob[14], prob[15]))
      
  print("Accuracy_01: %f, Accuracy_02: %f, Accuracy_03: %f, Accuracy_04: %f" % (accuracy_01, accuracy_02, accuracy_03, accuracy_04))
  print("Accuracy_05: %f, Accuracy_06: %f, Accuracy_07: %f, Accuracy_08: %f" % (accuracy_05, accuracy_06, accuracy_07, accuracy_08))
  print("Accuracy_09: %f, Accuracy_10: %f, Accuracy_11: %f, Accuracy_12: %f" % (accuracy_09, accuracy_10, accuracy_11, accuracy_12))
  print("Accuracy_13: %f, Accuracy_14: %f, Accuracy_15: %f, Accuracy_16: %f" % (accuracy_13, accuracy_14, accuracy_15, accuracy_16))
  
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
plt.plot(total_loss_09.numpy())
plt.plot(total_loss_10.numpy())
plt.plot(total_loss_11.numpy())
plt.plot(total_loss_12.numpy())
plt.plot(total_loss_13.numpy())
plt.plot(total_loss_14.numpy())
plt.plot(total_loss_15.numpy())
plt.plot(total_loss_16.numpy())
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
plt.plot(total_sparsity_09.numpy())
plt.plot(total_sparsity_10.numpy())
plt.plot(total_sparsity_11.numpy())
plt.plot(total_sparsity_12.numpy())
plt.plot(total_sparsity_13.numpy())
plt.plot(total_sparsity_14.numpy())
plt.plot(total_sparsity_15.numpy())
plt.plot(total_sparsity_16.numpy())
plt.axis([0,opt.n_epochs,0,cnt])
plt.show()

print("\n")
print('Rememver to save best one in %d models' % opt.ensemble)

'''
torch.save({'epoch': opt.n_epochs,
            'model_state_dict': model_01.state_dict(),
            'optimizer_state_dict': optimizer_01.state_dict(),
            'loss': loss_01}, 'C:/유민형/개인 연구/model compression/models/best_1.3.5.pkl')


for i in range(opt.layer):
   dist1 = 1-cal_sparsity(params_01[i].data).item()/params_01[i].data.reshape(-1,).shape[0]
   dist2 = 1-cal_sparsity(params_02[i].data).item()/params_02[i].data.reshape(-1,).shape[0]
   dist3 = 1-cal_sparsity(params_03[i].data).item()/params_03[i].data.reshape(-1,).shape[0]
   dist4 = 1-cal_sparsity(params_04[i].data).item()/params_04[i].data.reshape(-1,).shape[0]
   dist5 = 1-cal_sparsity(params_05[i].data).item()/params_05[i].data.reshape(-1,).shape[0]
   dist6 = 1-cal_sparsity(params_06[i].data).item()/params_06[i].data.reshape(-1,).shape[0]
   dist7 = 1-cal_sparsity(params_07[i].data).item()/params_07[i].data.reshape(-1,).shape[0]
   dist8 = 1-cal_sparsity(params_08[i].data).item()/params_08[i].data.reshape(-1,).shape[0]
   dist9 = 1-cal_sparsity(params_09[i].data).item()/params_09[i].data.reshape(-1,).shape[0]
   dist10 = 1-cal_sparsity(params_10[i].data).item()/params_10[i].data.reshape(-1,).shape[0]
   dist11 = 1-cal_sparsity(params_11[i].data).item()/params_11[i].data.reshape(-1,).shape[0]
   dist12 = 1-cal_sparsity(params_12[i].data).item()/params_12[i].data.reshape(-1,).shape[0]
   dist13 = 1-cal_sparsity(params_13[i].data).item()/params_13[i].data.reshape(-1,).shape[0]
   dist14 = 1-cal_sparsity(params_14[i].data).item()/params_14[i].data.reshape(-1,).shape[0]
   dist15 = 1-cal_sparsity(params_15[i].data).item()/params_15[i].data.reshape(-1,).shape[0]
   dist16 = 1-cal_sparsity(params_16[i].data).item()/params_16[i].data.reshape(-1,).shape[0]
   
   avg = (dist1+dist2+dist3+dist4+dist5+dist6+dist7+dist8+dist9+dist10+dist11+dist12+dist13+dist14+dist15+dist16)/16  
   
   print("Mean of sparsity in #%d layer: %f" % (i, avg))
    
'''




















