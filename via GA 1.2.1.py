import numpy as np
#from urllib import request
#import gzip
import pickle

'''
filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

if __name__ == '__main__':
    init()
'''

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#### Load data ####
train_x, train_y, test_x, test_y = load()

train_x = train_x.reshape(train_x.shape[0],1,28,28)/255
test_x = test_x.reshape(test_x.shape[0],1,28,28)/255

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
#test_y = torch.from_numpy(test_y)

train = TensorDataset(train_x, train_y)
train_loader = DataLoader(train, batch_size = 128, shuffle = True)

print(len(train_loader)) # train sample 나누기 batch size

#### Architecture: Lenet-5 ####
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    
    self.conv1 = nn.Conv2d(1, 6, 5, padding=2) # 0
    self.conv3 = nn.Conv2d(6, 16, 5) # 2
    self.conv5 = nn.Conv2d(16, 120, 5) # 4
    
    self.fc6 = nn.Linear(120, 84) # 6
    self.fc7 = nn.Linear(84, 10) # 8
    
  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv3(x)), 2)
    x = F.relu(self.conv5(x))
    x = x.view(-1, 120)
    x = F.relu(self.fc6(x))
    x = self.fc7(x)
    return F.log_softmax(x, dim=0)

#### Loss function ####
criterion = nn.CrossEntropyLoss()

#### Models ####
model_01 = Net().cuda()
optimizer_01 = optim.SGD(model_01.parameters(), lr=0.001)
params_01 = list(model_01.parameters())

model_02 = Net().cuda()
optimizer_02 = optim.SGD(model_02.parameters(), lr=0.001)
params_02 = list(model_02.parameters())

model_03 = Net().cuda()
optimizer_03 = optim.SGD(model_03.parameters(), lr=0.001)
params_03 = list(model_03.parameters())

model_04 = Net().cuda()
optimizer_04 = optim.SGD(model_04.parameters(), lr=0.001)
params_04 = list(model_04.parameters())

model_05 = Net().cuda()
optimizer_05 = optim.SGD(model_05.parameters(), lr=0.001)
params_05 = list(model_05.parameters())

model_06 = Net().cuda()
optimizer_06 = optim.SGD(model_06.parameters(), lr=0.001)
params_06 = list(model_06.parameters())

model_07 = Net().cuda()
optimizer_07 = optim.SGD(model_07.parameters(), lr=0.001)
params_07 = list(model_07.parameters())

model_08 = Net().cuda()
optimizer_08 = optim.SGD(model_08.parameters(), lr=0.001)
params_08 = list(model_08.parameters())

cnt = 0
for i in range(10):
   cnt += params_01[i].data.reshape(-1,).shape[0]   
print("How many total parameters: %d" % cnt)

#### initial mask ####
torch.manual_seed(1)
mask_01 = torch.FloatTensor(params_01[2].data.shape).uniform_() > 0.1
torch.manual_seed(2)
mask_02 = torch.FloatTensor(params_02[2].data.shape).uniform_() > 0.1
torch.manual_seed(3)
mask_03 = torch.FloatTensor(params_03[2].data.shape).uniform_() > 0.1
torch.manual_seed(4)
mask_04 = torch.FloatTensor(params_04[2].data.shape).uniform_() > 0.1
torch.manual_seed(5)
mask_05 = torch.FloatTensor(params_05[2].data.shape).uniform_() > 0.1
torch.manual_seed(6)
mask_06 = torch.FloatTensor(params_06[2].data.shape).uniform_() > 0.1
torch.manual_seed(7)
mask_07 = torch.FloatTensor(params_07[2].data.shape).uniform_() > 0.1
torch.manual_seed(8)
mask_08 = torch.FloatTensor(params_08[2].data.shape).uniform_() > 0.1

mask_01 = mask_01.float().cuda()
mask_02 = mask_02.float().cuda()
mask_03 = mask_03.float().cuda()
mask_04 = mask_04.float().cuda()
mask_05 = mask_05.float().cuda()
mask_06 = mask_06.float().cuda()
mask_07 = mask_07.float().cuda()
mask_08 = mask_08.float().cuda()

#### Functions ####
def cal_sparsity(*tensors): # Masking Tensor에 0이 몇 개 있는지 세는 함수
  result = 0
  for tensor in tensors:
    result += (tensor == 0).sum()
  return result

def minmax_sparsity(*sparsities): # 여러 모델의 sparsity를 minmax화 하는 함수
  sparsities = torch.Tensor(sparsities)
  if sparsities.max() == sparsities.min():
    minmax = torch.zeros((sparsities.shape))
  else:
    minmax = torch.empty((sparsities.shape))
    min_sparse = sparsities.min()
    bunmo = sparsities.max()-min_sparse
    for idx, sparse in enumerate(sparsities):
      minmax[idx] = (sparse-min_sparse)/bunmo
  return minmax

def minmax_recip_loss(*losses): # 여러 모델의 Loss를 minmax화 하는 함수
  losses = torch.Tensor(losses)
  reciprocal = torch.empty((losses.shape))
  for idx, loss in enumerate(losses):
    reciprocal[idx] = 1/loss
  if reciprocal.max() == reciprocal.min():
    minmax = torch.zeros((reciprocal.shape))
  else:
    minmax = torch.empty((reciprocal.shape))
    min_rec = reciprocal.min()
    bunmo = reciprocal.max()-min_rec
    for idx, loss in enumerate(reciprocal):
      minmax[idx] = (loss-min_rec)/bunmo
  return minmax

def make_prob(minmax_Sparse, minmax_Loss): # minmax된 sparsity와 loss로 object function을 만들고 확률화 하는 함수
  total = minmax_Sparse + minmax_Loss
  result = total/total.sum()
  return result

def cross4D_over(mask1, mask2): # 4차원 교차 함수
  #assert mask1.dim() >= 4, "Dimension is not 4"
  #assert mask2.dim() >= 4, "Dimension is not 4"
  #assert mask1.dim() == mask2.dim(), "Dimension is not equal"
  
  dim = mask1.dim()
  limit = 2 if mask1.shape[dim-4] == 1 else mask1.shape[dim-4]
  which = torch.LongTensor(1).random_(1, limit)[0] 
  
  copy = mask1[0:which,:,:,:].clone().detach()
  mask1[0:which,:,:,:] = mask2[0:which,:,:,:]
  mask2[0:which,:,:,:] = copy
  
def cross3D_over(mask1, mask2): # 3차원 교차 함수
  #assert mask1.dim() >= 3, "Dimension is not 3"
  #assert mask2.dim() >= 3, "Dimension is not 3"
  #assert mask1.dim() == mask2.dim(), "Dimension is not equal"
  
  dim = mask1.dim()
  limit2 = 2 if mask1.shape[dim-3] == 1 else mask1.shape[dim-3]
  which2 = torch.LongTensor(1).random_(1, limit2)[0] 
  
  if dim == 4:
    limit = 2 if mask1.shape[dim-4] == 1 else mask1.shape[dim-4]
    which = torch.LongTensor(1).random_(1, limit)[0]
    
    copy = mask1[0:which,0:which2,:,:].clone().detach()
    mask1[0:which,0:which2,:,:] = mask2[0:which,0:which2,:,:]
    mask2[0:which,0:which2,:,:] = copy
  else:
    copy = mask1[0:which2,:,:].clone().detach()
    mask1[0:which2,:,:] = mask2[0:which2,:,:]
    mask2[0:which2,:,:] = copy

def cross2D_over(mask1, mask2): # 2차원 교차 함수
  #assert mask1.dim() >= 2, "Dimension is not 2"
  #assert mask2.dim() >= 2, "Dimension is not 2"
  #assert mask1.dim() == mask2.dim(), "Dimension is not equal"
  
  dim = mask1.dim()
  limit3 = 2 if mask1.shape[dim-2] == 1 else mask1.shape[dim-2]
  which3 = torch.LongTensor(1).random_(1, limit3)[0]
  
  if dim == 4:
    limit = 2 if mask1.shape[dim-4] == 1 else mask1.shape[dim-4]
    limit2 = 2 if mask1.shape[dim-3] == 1 else mask1.shape[dim-3]
    which = torch.LongTensor(1).random_(1, limit)[0]
    which2 = torch.LongTensor(1).random_(1, limit2)[0]
    
    copy = mask1[0:which,0:which2,0:which3,:].clone().detach()
    mask1[0:which,0:which2,0:which3,:] = mask2[0:which,0:which2,0:which3,:]
    mask2[0:which,0:which2,0:which3,:] = copy
  elif dim == 3:
    limit2 = 2 if mask1.shape[dim-3] == 1 else mask1.shape[dim-3]
    which2 = torch.LongTensor(1).random_(1, limit2)[0]
    
    copy = mask1[0:which2,0:which3,:].clone().detach()
    mask1[0:which2,0:which3,:] = mask2[0:which2,0:which3,:]
    mask2[0:which2,0:which3,:] = copy
  else:
    copy = mask1[0:which3,:].clone().detach()
    mask1[0:which3,:] = mask2[0:which3,:]
    mask2[0:which3,:] = copy
    
def cross1D_over(mask1, mask2): # 1차원 교차 함수
  #assert mask1.dim() >= 1, "Dimension is not 1"
  #assert mask2.dim() >= 1, "Dimension is not 1"
  #assert mask1.dim() == mask2.dim(), "Dimension is not equal"
  
  dim = mask1.dim()
  limit4 = 2 if mask1.shape[dim-1] == 1 else mask1.shape[dim-1]
  which4 = torch.LongTensor(1).random_(1, limit4)[0]
  
  if dim == 4:
    limit = 2 if mask1.shape[dim-4] == 1 else mask1.shape[dim-4]
    limit2 = 2 if mask1.shape[dim-3] == 1 else mask1.shape[dim-3]
    limit3 = 2 if mask1.shape[dim-2] == 1 else mask1.shape[dim-2]
    which = torch.LongTensor(1).random_(1, limit)[0]
    which2 = torch.LongTensor(1).random_(1, limit2)[0]
    which3 = torch.LongTensor(1).random_(1, limit3)[0]
    
    copy = mask1[0:which,0:which2,0:which3,0:which4].clone().detach()
    mask1[0:which,0:which2,0:which3,0:which4] = mask2[0:which,0:which2,0:which3,0:which4]
    mask2[0:which,0:which2,0:which3,0:which4] = copy
  elif dim == 3:
    limit2 = 2 if mask1.shape[dim-3] == 1 else mask1.shape[dim-3]
    limit3 = 2 if mask1.shape[dim-2] == 1 else mask1.shape[dim-2]
    which2 = torch.LongTensor(1).random_(1, limit2)[0]
    which3 = torch.LongTensor(1).random_(1, limit3)[0]
    
    copy = mask1[0:which2,0:which3,0:which4].clone().detach()
    mask1[0:which2,0:which3,0:which4] = mask2[0:which2,0:which3,0:which4]
    mask2[0:which2,0:which3,0:which4] = copy
  elif dim == 2:
    limit3 = 2 if mask1.shape[dim-2] == 1 else mask1.shape[dim-2]
    which3 = torch.LongTensor(1).random_(1, limit3)[0]
    
    copy = mask1[0:which3,0:which4].clone().detach()
    mask1[0:which3,0:which4] = mask2[0:which3,0:which4]
    mask2[0:which3,0:which4] = copy
  else:
    copy = mask1[0:which4].clone().detach()
    mask1[0:which4] = mask2[0:which4]
    mask2[0:which4] = copy

def cross_over(mask1,mask2): # 교차 함수를 랜덤하게 선택하여 사용하는 함수
  assert mask1.dim() == mask2.dim(), "Dimension is not equal"
  
  dim = mask1.dim()
  if dim == 1:
    cross1D_over(mask1, mask2)
  elif dim == 2:
    which = torch.LongTensor(1).random_(1, 3)[0]
    if which == 1:
      cross1D_over(mask1, mask2)
    else:
      cross2D_over(mask1, mask2)
  elif dim == 3:
    which = torch.LongTensor(1).random_(1, 4)[0]
    if which == 1:
      cross1D_over(mask1, mask2)
    elif which == 2:
      cross2D_over(mask1, mask2)
    else:
      cross3D_over(mask1, mask2)
  else:
    which = torch.LongTensor(1).random_(1, 5)[0]
    if which == 1:
      cross1D_over(mask1, mask2)
    elif which == 2:
      cross2D_over(mask1, mask2)
    elif which == 3:
      cross3D_over(mask1, mask2)
    else:
      cross4D_over(mask1, mask2)

def Mutate4D(mask):
    
  dim = mask.dim()
  limit = 2 if mask.shape[dim-4] == 1 else mask.shape[dim-4] + 1
  which = torch.multinomial(torch.ones(limit)*0.25, 2, replacement=False)
  
  copy = 1 - mask[which.min():which.max(),:,:,:].clone().detach()
  mask[which.min():which.max(),:,:,:] = copy

def Mutate3D(mask):
    
  dim = mask.dim()
  limit2 = 2 if mask.shape[dim-3] == 1 else mask.shape[dim-3] + 1
  which2 = torch.multinomial(torch.ones(limit2)*0.25, 2, replacement=False)
  
  if dim == 4:
    limit = 2 if mask.shape[dim-4] == 1 else mask.shape[dim-4] + 1
    which = torch.multinomial(torch.ones(limit)*0.25, 2, replacement=False)
    
    copy = 1 - mask[which.min():which.max(),which2.min():which2.max(),:,:].clone().detach()
    mask[which.min():which.max(),which2.min():which2.max(),:,:] = copy
    
  else:
    copy = 1 - mask[which2.min():which2.max(),:,:].clone().detach()
    mask[which2.min():which2.max(),:,:] = copy

def Mutate2D(mask):
  
  dim = mask.dim()
  limit3 = 2 if mask.shape[dim-2] == 1 else mask.shape[dim-2] + 1
  which3 = torch.multinomial(torch.ones(limit3)*0.25, 2, replacement=False)
  
  if dim == 4:
    limit = 2 if mask.shape[dim-4] == 1 else mask.shape[dim-4] + 1
    limit2 = 2 if mask.shape[dim-3] == 1 else mask.shape[dim-3] + 1
    which = torch.multinomial(torch.ones(limit)*0.25, 2, replacement=False)
    which2 = torch.multinomial(torch.ones(limit2)*0.25, 2, replacement=False)
    
    copy = 1 - mask[which.min():which.max(),which2.min():which2.max(),which3.min():which3.max(),:].clone().detach()
    mask[which.min():which.max(),which2.min():which2.max(),which3.min():which3.max(),:] = copy

  elif dim == 3:
    limit2 = 2 if mask.shape[dim-3] == 1 else mask.shape[dim-3] + 1
    which2 = torch.multinomial(torch.ones(limit2)*0.25, 2, replacement=False)
    
    copy = 1 - mask[which2.min():which2.max(),which3.min():which3.max(),:].clone().detach()
    mask[which2.min():which2.max(),which3.min():which3.max(),:] = copy
  
  else:
    copy = 1 - mask[which3.min():which3.max(),:].clone().detach()
    mask[which3.min():which3.max(),:] = copy

def Mutate1D(mask):
    
  dim = mask.dim()
  limit4 = 2 if mask.shape[dim-1] == 1 else mask.shape[dim-1] + 1
  which4 = torch.multinomial(torch.ones(limit4)*0.25, 2, replacement=False)
  
  if dim == 4:
    limit = 2 if mask.shape[dim-4] == 1 else mask.shape[dim-4] + 1
    limit2 = 2 if mask.shape[dim-3] == 1 else mask.shape[dim-3] + 1
    limit3 = 2 if mask.shape[dim-2] == 1 else mask.shape[dim-2] + 1
    which = torch.multinomial(torch.ones(limit)*0.25, 2, replacement=False)
    which2 = torch.multinomial(torch.ones(limit2)*0.25, 2, replacement=False)
    which3 = torch.multinomial(torch.ones(limit3)*0.25, 2, replacement=False)
    
    copy = 1 - mask[which.min():which.max(),which2.min():which2.max(),which3.min():which3.max(),which4.min():which4.max()].clone().detach()
    mask[which.min():which.max(),which2.min():which2.max(),which3.min():which3.max(),which4.min():which4.max()] = copy
    
  elif dim == 3:
    limit2 = 2 if mask.shape[dim-3] == 1 else mask.shape[dim-3] + 1
    limit3 = 2 if mask.shape[dim-2] == 1 else mask.shape[dim-2] + 1
    which2 = torch.multinomial(torch.ones(limit2)*0.25, 2, replacement=False)
    which3 = torch.multinomial(torch.ones(limit3)*0.25, 2, replacement=False)
    
    copy = 1 - mask[which2.min():which2.max(),which3.min():which3.max(),which4.min():which4.max()].clone().detach()
    mask[which2.min():which2.max(),which3.min():which3.max(),which4.min():which4.max()] = copy
  
  elif dim == 2:
    limit3 = 2 if mask.shape[dim-2] == 1 else mask.shape[dim-2] + 1
    which3 = torch.multinomial(torch.ones(limit3)*0.25, 2, replacement=False)
    
    copy = 1 - mask[which3.min():which3.max(),which4.min():which4.max()].clone().detach()
    mask[which3.min():which3.max(),which4.min():which4.max()] = copy
  
  else:
    copy = 1 - mask[which4.min():which4.max()].clone().detach()
    mask[which4.min():which4.max()] = copy

def decision(prob): # 확률에 따른 결정을 내리는 함수
  return torch.FloatTensor(1).uniform_(0,1)[0] < prob

def Mutate(mask, prob):
  if decision(prob):
    print("Mutation!")
    dim = mask.dim()
    if dim == 1:
      Mutate1D(mask)
    elif dim == 2:
      which = torch.LongTensor(1).random_(1, 3)[0]
      if which == 1:
        Mutate1D(mask)
      else:
        Mutate2D(mask)
    elif dim == 3:
      which = torch.LongTensor(1).random_(1, 4)[0]
      if which == 1:
        Mutate1D(mask)
      elif which == 2:
        Mutate2D(mask)
      else:
        Mutate3D(mask)
    else:
      which = torch.LongTensor(1).random_(1, 5)[0]
      if which == 1:
        Mutate1D(mask)
      elif which == 2:
        Mutate2D(mask)
      elif which == 3:
        Mutate3D(mask)
      else:
        Mutate4D(mask)

###################################### 실험 #########################################################################
###################################### different initialize, not sharing parameters #################################

n_epochs = 100 # 학습 횟수
alpha = 1
total_loss_01 = torch.zeros((n_epochs))
total_loss_02 = torch.zeros((n_epochs))
total_loss_03 = torch.zeros((n_epochs))
total_loss_04 = torch.zeros((n_epochs))
total_loss_05 = torch.zeros((n_epochs))
total_loss_06 = torch.zeros((n_epochs))
total_loss_07 = torch.zeros((n_epochs))
total_loss_08 = torch.zeros((n_epochs))

for epoch in range(n_epochs):
  
  #### weight removing ####
  params_01[2].data = params_01[2].data * mask_01 
  params_02[2].data = params_02[2].data * mask_02 
  params_03[2].data = params_03[2].data * mask_03 
  params_04[2].data = params_04[2].data * mask_04 
  params_05[2].data = params_05[2].data * mask_05
  params_06[2].data = params_06[2].data * mask_06 
  params_07[2].data = params_07[2].data * mask_07 
  params_08[2].data = params_08[2].data * mask_08 
  
  #### model training ####
  for batch_idx, (x, y) in enumerate(train_loader):
    
    x = Variable(x).float().cuda()
    y = Variable(y).long().cuda() 
    
    ## model_01 ##
    optimizer_01.zero_grad()
    output_01 = model_01(x)
    loss_01 = criterion(output_01, y)        
    loss_01.backward()
    params_01[2].grad = params_01[2].grad * mask_01 # gradient removing
    optimizer_01.step()
    total_loss_01[epoch] += loss_01.item()
    
    ## model_02 ##
    optimizer_02.zero_grad()
    output_02 = model_02(x)
    loss_02 = criterion(output_02, y)        
    loss_02.backward()
    params_02[2].grad = params_02[2].grad * mask_02 # gradient removing
    optimizer_02.step()
    total_loss_02[epoch] += loss_02.item()
    
    ## model_03 ##
    optimizer_03.zero_grad()
    output_03 = model_03(x)
    loss_03 = criterion(output_03, y)        
    loss_03.backward()
    params_03[2].grad = params_03[2].grad * mask_03 # gradient removing
    optimizer_03.step()
    total_loss_03[epoch] += loss_03.item()
    
    ## model_04 ##
    optimizer_04.zero_grad()
    output_04 = model_04(x)
    loss_04 = criterion(output_04, y)        
    loss_04.backward()
    params_04[2].grad = params_04[2].grad * mask_04 # gradient removing
    optimizer_04.step()
    total_loss_04[epoch] += loss_04.item()
    
    ## model_05 ##
    optimizer_05.zero_grad()
    output_05 = model_05(x)
    loss_05 = criterion(output_05, y)        
    loss_05.backward()
    params_05[2].grad = params_05[2].grad * mask_05 # gradient removing
    optimizer_05.step()
    total_loss_05[epoch] += loss_05.item()
    
    ## model_06 ##
    optimizer_06.zero_grad()
    output_06 = model_06(x)
    loss_06 = criterion(output_06, y)        
    loss_06.backward()
    params_06[2].grad = params_06[2].grad * mask_06 # gradient removing
    optimizer_06.step()
    total_loss_06[epoch] += loss_06.item()
    
    ## model_07 ##
    optimizer_07.zero_grad()
    output_07 = model_07(x)
    loss_07 = criterion(output_07, y)        
    loss_07.backward()
    params_07[2].grad = params_07[2].grad * mask_07 # gradient removing
    optimizer_07.step()
    total_loss_07[epoch] += loss_07.item()
    
    ## model_08 ##
    optimizer_08.zero_grad()
    output_08 = model_08(x)
    loss_08 = criterion(output_08, y)        
    loss_08.backward()
    params_08[2].grad = params_08[2].grad * mask_08 # gradient removing
    optimizer_08.step()
    total_loss_08[epoch] += loss_08.item()
  
  total_loss_01[epoch] /= len(train_loader)
  total_loss_02[epoch] /= len(train_loader)
  total_loss_03[epoch] /= len(train_loader)
  total_loss_04[epoch] /= len(train_loader)
  total_loss_05[epoch] /= len(train_loader)
  total_loss_06[epoch] /= len(train_loader)
  total_loss_07[epoch] /= len(train_loader)
  total_loss_08[epoch] /= len(train_loader)
  
  #### masking selection ####
  
  ## get fitness ##
  sparsity_01 = cal_sparsity(mask_01)
  sparsity_02 = cal_sparsity(mask_02)
  sparsity_03 = cal_sparsity(mask_03)
  sparsity_04 = cal_sparsity(mask_04)
  sparsity_05 = cal_sparsity(mask_05)
  sparsity_06 = cal_sparsity(mask_06)
  sparsity_07 = cal_sparsity(mask_07)
  sparsity_08 = cal_sparsity(mask_08)
  
  S = minmax_sparsity(sparsity_01, sparsity_02, sparsity_03, sparsity_04, sparsity_05, sparsity_06, sparsity_07, sparsity_08)
  L = minmax_recip_loss(total_loss_01[epoch],total_loss_02[epoch],total_loss_03[epoch],total_loss_04[epoch],
                        total_loss_05[epoch],total_loss_06[epoch],total_loss_07[epoch],total_loss_08[epoch])
  prob = make_prob(alpha*S,L)
  
  ## selection ##
  copy_01 = torch.empty((mask_01.shape))
  copy_02 = torch.empty((mask_02.shape))
  copy_03 = torch.empty((mask_03.shape))
  copy_04 = torch.empty((mask_04.shape))
  copy_05 = torch.empty((mask_05.shape))
  copy_06 = torch.empty((mask_06.shape))
  copy_07 = torch.empty((mask_07.shape))
  copy_08 = torch.empty((mask_08.shape))
  
  copy = [copy_01, copy_02, copy_03, copy_04, copy_05, copy_06, copy_07, copy_08]
  mask_list = [mask_01, mask_02, mask_03, mask_04, mask_05, mask_06, mask_07, mask_08]
  
  for smp in range(8):
    #copy[smp] = mask_list[torch.multinomial(prob,1)[0]].clone().detach()
    copy[smp] = mask_list[np.random.choice(8, 1, p=prob.numpy())[0]].clone().detach()
  
  ## simple pairs : (copy_01, copy_02), (copy_03, copy_04) ##
  ## cross over ##
  cross_over(copy[0],copy[1])
  cross_over(copy[2],copy[3])
  cross_over(copy[4],copy[5])
  cross_over(copy[6],copy[7])
  
  ## mutation ##
  Mutate(copy[0],0.1)
  Mutate(copy[1],0.1)
  Mutate(copy[2],0.1)
  Mutate(copy[3],0.1)
  Mutate(copy[4],0.1)
  Mutate(copy[5],0.1)
  Mutate(copy[6],0.1)
  Mutate(copy[7],0.1)
  
  ## inherit ##
  mask_01 = copy[0].clone().detach().float().cuda()
  mask_02 = copy[1].clone().detach().float().cuda()
  mask_03 = copy[2].clone().detach().float().cuda()
  mask_04 = copy[3].clone().detach().float().cuda()
  mask_05 = copy[4].clone().detach().float().cuda()
  mask_06 = copy[5].clone().detach().float().cuda()
  mask_07 = copy[6].clone().detach().float().cuda()
  mask_08 = copy[7].clone().detach().float().cuda()
  
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
  
  #### print ####
  
  print("=" * 100)
  print("Epoch: %d" % (epoch+1))
  print("Loss_01: %f, Loss_02: %f, Loss_03: %f, Loss_04: %f" % (total_loss_01[epoch],total_loss_02[epoch],total_loss_03[epoch],total_loss_04[epoch]))
  print("Loss_05: %f, Loss_06: %f, Loss_07: %f, Loss_08: %f" % (total_loss_05[epoch],total_loss_06[epoch],total_loss_07[epoch],total_loss_08[epoch]))
  print("Sparsity_01: %d, Sparsity_02: %d, Sparsity_03: %d, Sparsity_04: %d" % (sparsity_01, sparsity_02, sparsity_03, sparsity_04))
  print("Sparsity_05: %d, Sparsity_06: %d, Sparsity_07: %d, Sparsity_08: %d" % (sparsity_05, sparsity_06, sparsity_07, sparsity_08))
  print("Prob_01: %f, Prob_02: %f, Prob_03: %f, Prob_04: %f" % (prob[0], prob[1], prob[2], prob[3]))
  print("Prob_05: %f, Prob_06: %f, Prob_07: %f, Prob_08: %f" % (prob[4], prob[5], prob[6], prob[7]))
  print("Accuracy_01: %f, Accuracy_02: %f, Accuracy_03: %f, Accuracy_04: %f" % (accuracy_01, accuracy_02, accuracy_03, accuracy_04))
  print("Accuracy_05: %f, Accuracy_06: %f, Accuracy_07: %f, Accuracy_08: %f" % (accuracy_05, accuracy_06, accuracy_07, accuracy_08))
  
  
  


































