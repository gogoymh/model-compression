import csv
import numpy as np
import ast


f = open("C:/유민형/개인 연구/model compression/results/1.7.9.4/3_Result_1.7.9.4.tsv",'r', encoding='utf-8', newline='')
lines = csv.reader(f, delimiter='\t')
a = [line for line in lines]
                    
sparsity_tmp = np.empty((16,3))
loss_tmp = np.empty((16,3))

for j in range(16):
    sparsity_tmp[j] = np.array(ast.literal_eval(a[1][j]))
    loss_tmp[j] = np.array(ast.literal_eval(a[3][j]))