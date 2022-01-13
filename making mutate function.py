# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:11:35 2019

@author: Minhyeong
"""
import torch

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


def Mutate(mask, prob):
  if decision(prob):
    
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