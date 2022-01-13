import torch
import pickle

#### Functions ####
def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

def minmax_fn(values): # 여러 모델의 sparsity를 minmax화 하는 함수
  #values = torch.Tensor(values)
  values = values.float()
  if values.max() == values.min():
    minmax = torch.zeros((values.shape))
  else:
    minmax = torch.empty((values.shape))
    min_sparse = values.min()
    bunmo = values.max()-min_sparse
    for idx, sparse in enumerate(values):
      minmax[idx] = (sparse-min_sparse)/bunmo
  return minmax

def cal_sparsity(*tensors): # Masking Tensor에 0이 몇 개 있는지 세는 함수
  result = 0
  for tensor in tensors:
    result += (tensor == 0).sum()
  return result

def cal_ch(params,changable,layer):
    channel_bonus = 0
    parallel_bonus = 0
    vertical = 0
    horizontal = 0
    for idx in range(len(changable)):
        i = changable[idx]
        if len(params[i].data.shape) == 2: # Linear kernel node pruning
            for k in range(params[i].data.shape[0]): # output channel or node
                if params[i].data[k,0].item() != 0:
                    continue
                elif (params[i].data[k,:] != 0).sum().item() == 0:
                    channel_bonus += 1
                    limit = changable[idx+1] if (idx+1) != len(changable) else layer # opt.layer
                    for j in range(i+1,limit):
                        if params[j].data[k] == 0:
                            parallel_bonus += 1
            
        else: # Convolutional kenel channel pruning             
            for k in range(params[i].data.shape[0]): # output channel or node
                if params[i].data[k,0,0,0].item() != 0:
                    continue
                elif (params[i].data[k,:,:,:] != 0).sum().item() == 0:
                    channel_bonus += 1
                    limit = changable[idx+1] if (idx+1) != len(changable) else layer # opt.layer
                    for j in range(i+1,limit):
                        if params[j].data[k] == 0:
                            parallel_bonus += 1
            for h in range(params[i].data.shape[2]):
                if (params[i].data[:,:,h,:] != 0).sum().item() == 0:
                    horizontal += 1
            for v in range(params[i].data.shape[3]):
                if (params[i].data[:,:,:,v] != 0).sum().item() == 0:
                    vertical += 1
                
        rectangular_bonus = vertical + horizontal
    return channel_bonus, parallel_bonus, rectangular_bonus

def make_prob(minmax_Sparse, minmax_Loss, ensemble): # minmax된 sparsity와 loss로 object function을 만들고 확률화 하는 함수
  total = minmax_Sparse + minmax_Loss
  result = total/total.sum() if total.sum() != 0 else torch.ones((ensemble))*(1/ensemble)
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

def Mutate(mask, prob, layer, ensemble):
  if decision(prob):
    print("Mutation! [Layer %2d] | [Model %2d]" % (layer, (ensemble+1)))
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
