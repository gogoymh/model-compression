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
from network import resnet18 as Net
from utils import cal_sparsity, cal_ch, make_prob, cross_over, Mutate, minmax_fn, decision

#################################################################################################################
model = Net()

params = list(model.parameters())

device = torch.device("cuda:0")

changable = []
for i in range(len(params)):
    if len(params[i].data.shape) == 4:
        changable.append(i)
print(changable)

#################################################################################################################
cnt = 0
changable_dim = []
for i in changable:
    cnt += params[i].data.reshape(-1,).shape[0]
    changable_dim.append(params[i].data.reshape(-1,).shape[0])
print(cnt)
print(changable_dim)

masks = []

for j in range(16):
    mask = []
    for i in range(62):
        mask_layer = torch.ones((params[i].data.shape)).float().to(device)
        mask.append(mask_layer)            
        
    masks.append(mask)

chromosomes = []
for j in range(16):
    chromosome = torch.ones((cnt))#.float().to(device)
    chromosomes.append(chromosome)

#################################################################################################################
# Genetic Algorithm
    
def selection(chromosomes, fitness, ensemble, elitism=True):
    new_chromosomes = []
    prob = make_prob(fitness)
    
    for j in range(ensemble):
        new_chromosomes.append(chromosomes[np.random.choice(ensemble, 1, p=prob.numpy())[0]])
    
    if elitism:
        new_chromosomes[0] = chromosomes[prob.argmax()]
    
    return new_chromosomes

def crossover(chromosomes, ensemble, cnt, probability=0.1):
    for k in range(ensemble/2):
        if decision(probability):
            point = np.random.choice(cnt,1)
            copy = chromosomes[2*k][0:point].clone().detach()
            chromosomes[2*k][0:point] = chromosomes[2*k+1][0:point]
            chromosomes[2*k+1][0:point] = copy
            print("Cross-over between %d and %d" % (2*k, 2*k+1))

def mutation(chromosomes, ensemble, cnt, probability=0.1):
    for j in range(ensemble):
        if decision(probability):
            point = np.random.choice(cnt,1)
            chromosomes[j][point] = 1 - chromosomes[j][point]
            print("Mutation in model %d" % (j+1))

def chromo2mask(chromosomes, masks, changable, changable_dim, ensemble):
    for j in range(ensemble):
        for i in range(len(changable)):
            if i == 0:
                masks[j][changable[0]] = chromosomes[j][0:changable_dim[i]].reshape(masks[j][changable[0]].shape).clone().detach().float().to(device)
            else:
                masks[j][changable[i]] = chromosomes[j][changable_dim[i-1]:changable_dim[i-1]+changable_dim[i]].reshape(masks[j][changable[i]].shape).clone().detach().float().to(device)
            


def parallel_bonus(masks, changable, ensemble, layer):
    channel_prune_bonus = torch.zeros((ensemble))
    for j in range(ensemble):
        for i in range(len(changable)):
            idx = changable[i]
            for k in range(masks[j][idx].shape[0]): # output channel or node
                if masks[j][idx].data[k,0,0,0].item() != 0:
                    continue
                elif (masks[j][idx][k,:,:,:] != 0).sum().item() == 0:
                    channel_prune_bonus[j] += 1
                    limit = changable[idx+1] if (idx+1) != len(changable) else layer # opt.layer
                    for l in range(idx+1,limit):
                        masks[j][l][k] = 0
    
    return channel_prune_bonus

#################################################################################################################
# Evaluate



















