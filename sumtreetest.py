# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:04:30 2019

@author: Minhyeong
"""

import numpy as np
import math

capacity = 6
values = ['a','b','c','d','e','f']
priorities = [1,3,5,7,9,11]

buffer = np.zeros(capacity, dtype=object)

def isPowerOfTwo (x):
    return (x and (not(x & (x - 1))))

if isPowerOfTwo(capacity):
    tree = np.zeros(2 * capacity - 1)
    tree_type = True
else:
    tree = np.zeros(2 ** (math.ceil(math.log2(capacity)) + 1) - 1)
    tree_type = False

buffer_idx = 0
n_entries = 0

def _propagate(tree, index, change):
    parent = (index - 1) // 2

    tree[parent] += change

    if parent != 0:
        _propagate(tree, parent, change)


def update_priority(tree, index, new_priority):
    change = new_priority - tree[index]
    tree[index] = new_priority
    _propagate(tree, index, change)
    
    return tree

def add(buffer, buffer_idx, n_entries, tree, value, priority):
    
    buffer[buffer_idx] = value
        
    if tree_type:
        tree_idx = buffer_idx + capacity - 1
    else:
        tree_idx = buffer_idx + (2 ** math.ceil(math.log2(capacity))) - 1
        
    tree = update_priority(tree, tree_idx, priority)
        
    buffer_idx += 1
        
    if buffer_idx >= capacity: # First in, First out
        buffer_idx = 0            
        
    if n_entries < capacity:
        n_entries += 1

    return buffer, buffer_idx, n_entries, tree


for i in range(capacity):
    buffer, buffer_idx, n_entries, tree = add(buffer, buffer_idx, n_entries, tree, values[i], priorities[i])

print("="*30)
print('value is', values)
print('priority is', priorities)
print("="*30)
print('buffer is', buffer)
print('tree is', tree)
print('length of tree is %d' % len(tree))
print('Entries are %d' % n_entries)


def _retrieve(tree, idx, s):
    left = 2 * idx + 1
    right = left + 1

    if left >= len(tree):
        return idx

    if s <= tree[left]:
        return _retrieve(tree, left, s)
    else:
        return _retrieve(tree, right, s - tree[left])

def get(tree, tree_type, buffer, s):
    idx = _retrieve(tree,0, s)
        
    if tree_type:
        dataIdx = idx - capacity + 1
    else:
        dataIdx = idx - (2 ** math.ceil(math.log2(capacity))) + 1

    return idx, tree[idx], buffer[dataIdx]



def sample(tree, tree_type, buffer, n_entries, n):
        batch = []
        idxs = []
        segment = tree[0] / n
        priorities = []

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            print('s is %f' % s)
            (idx, p, data) = get(tree, tree_type, buffer, s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / tree[0]
        is_weight = np.power(n_entries * sampling_probabilities, -0.5)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

batch, idxs, is_weight = sample(tree, False, buffer, n_entries, 4)
print("="*30)
print('batch is', batch)
print('idxs is', idxs)
print('weight is', is_weight)




