import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import torch.backends.cudnn as cudnn
import os
import csv
import timeit
import matplotlib.pyplot as plt

#### Custom Library #################################################################################################
from functions import cal_sparsity, minmax_sparsity, minmax_recip_loss, make_prob, cross_over, Mutate
from network import LeNet as Net

#### Hyper parameter ################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--initcuda", type=int, default=0, help="Initial number of cuda")
parser.add_argument("--numcuda", type=int, default=8, help="How many cuda to use")
parser.add_argument("--ver1", type=int, default=1, help="Version info")
parser.add_argument("--ver2", type=int, default=7, help="Version info")
parser.add_argument("--ver3", type=int, default=6, help="Version info")
parser.add_argument("--ver4", type=int, default=0, help="Version info")
parser.add_argument("--n_epochs", type=int, default=3, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
parser.add_argument("--mask", type=float, default=0.1, help="Masking rate for sparsity tensor")
parser.add_argument("--mutate", type=float, default=0.01, help="Mutate probability")
parser.add_argument("--alpha", type=float, default=1, help="Ratio between loss and sparsity")
parser.add_argument("--buffer", type=int, default=1, help="Buffer for converge")
parser.add_argument("--recovery", type=float, default=0.0001, help="Recovery constant")
parser.add_argument("--remask", type=float, default=0.99, help="Recovery masking rate")
parser.add_argument("--ensemble", type=int, default=16, help="Number of models")
parser.add_argument("--layer", type=int, default=10, help="Number of layers")

opt = parser.parse_args()

print("=" * 100)
print("<<<<<<<<<<Double Check Hyper Parameter>>>>>>>>>>")
print("Experiment Version              | %d.%d.%d.%d" % (opt.ver1, opt.ver2, opt.ver3, opt.ver4))
print("Total Epoch is                  | %d" % opt.n_epochs)
print("Batch size is                   | %d" % opt.batch_size)
print("Leraning rate is                | %f" % opt.lr)
print("Initial Mask rate is            | %f" % opt.mask)
print("Mutation probability is         | %f" % opt.mutate)
print("Balance factor alpha is         | %f" % opt.alpha)
print("How long buffer is              | %d" % opt.buffer)
print("Recovery constant is            | %f" % opt.recovery)
print("Recovery masking is             | %f" % opt.remask)
print("How many models are ensembled   | %d" % opt.ensemble)
print("How many layers will be pruned  | %d" % opt.layer)

#### path ###########################################################################################################
main_path = 'C:/유민형/개인 연구/model compression/results/%d.%d.%d.%d' % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)

if not os.path.isdir(main_path):
    os.mkdir(main_path)

#### Load data ######################################################################################################
train_loader = DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(
                        (0.1307,), (0.3081,))]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True
)
        
test_loader = DataLoader(
        datasets.MNIST(
            '../data/mnist',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(
                        (0.1307,), (0.3081,))]
            ),
        ),
        batch_size=1000,
        shuffle=False
)

print("Length of train loader is       | %d" % len(train_loader)) # train sample 나누기 batch size

#### Loss function ##################################################################################################
criterion = nn.CrossEntropyLoss()

#### Models ####
models = []
optims = []
params = []

if torch.cuda.is_available():
    print("Model will be build on          | %d GPU" % torch.cuda.device_count())
    for c in range(torch.cuda.device_count()):
        print("The current device is           | cuda:%d " % torch.cuda.current_device())
else:
    print("Model will be build on          | CPU")

gpu_list = list(range(opt.initcuda, opt.initcuda + opt.numcuda))
for j in range(opt.ensemble):
    #torch.cuda.manual_seed(j)
    model = Net()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, gpu_list).cuda()
        cudnn.benchmark = True
    else:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    parameter = list(model.parameters())
    
    models.append(model)
    optims.append(optimizer)
    params.append(parameter)

#### Count Parameters ###############################################################################################
cnt = 0
for i in range(opt.layer):
   cnt += params[0][i].data.reshape(-1,).shape[0]   
print("How many total parameters       | %d" % cnt)

#### initial mask ###################################################################################################
masks = []

for j in range(opt.ensemble):
    mask = []
    for i in range(opt.layer):
        #torch.manual_seed(j)
        mask_layer = torch.FloatTensor(params[j][i].data.shape).uniform_() > opt.mask
        mask_layer = mask_layer.float().cuda()
        mask.append(mask_layer)
    masks.append(mask)

#### 실험 ###########################################################################################################
total_loss = torch.zeros((opt.ensemble, opt.n_epochs))

total_sparsity = torch.zeros((opt.ensemble, opt.n_epochs)).long()

##########################################################################################################################
start = timeit.default_timer()

for epoch in range(opt.n_epochs):
  print("=" * 100)
  print("Epoch: %d" % (epoch+1))  
  
  if epoch >= opt.buffer:
      #### model broken test ####
      for j in range(opt.ensemble):
          if cal_sparsity(params[j][0].grad).item() == params[0][0].data.reshape(-1,).shape[0] :
              collapsed = 0
              for i in range(2,9,2):
                  if cal_sparsity(params[j][i].grad).item() == params[0][i].data.reshape(-1,).shape[0] :
                      collapsed += 2
              if collapsed >= 2:
                  params[j][collapsed-2].data += opt.recovery * (params[j][collapsed-2].data == 0).float() * (torch.FloatTensor(params[j][collapsed-2].data.shape).uniform_() > opt.remask).float().cuda()
              params[j][collapsed].data += opt.recovery * (params[j][collapsed].data == 0).float() * (torch.FloatTensor(params[j][collapsed].data.shape).uniform_() > opt.remask).float().cuda()
              print("!" * 30)
              print("Layer %d of Model %d was collapsed. Get recovered!" % (collapsed, j+1))
              print("!" * 30)
              
      #### weight removing ####
      for i in range(opt.layer):
          for j in range(opt.ensemble):
              params[j][i].data *= masks[j][i]
  
  #### train ####
  for batch_idx, (x, y) in enumerate(train_loader):
      
    x = Variable(x).float().cuda()
    y = Variable(y).long().cuda() 
    
    for j in range(opt.ensemble):

        optims[j].zero_grad()        
        output = models[j](x)        
        loss = criterion(output, y)        
        loss.backward()
        
        if epoch >= opt.buffer:
            for i in range(opt.layer): # gradient removing
                params[j][i].grad *= masks[j][i]
        
        optims[j].step()        
        total_loss[j,epoch] += loss.item()
        
  for j in range(opt.ensemble):
      total_loss[j,epoch] /= len(train_loader)
  
  #### print ####  
  for k in range(opt.ensemble//4):
      print("Loss_%02d: %f, Loss_%02d: %f, Loss_%02d: %f, Loss_%02d: %f" % 
            ((4*k+1),total_loss[4*k,epoch],(4*k+2),total_loss[4*k+1,epoch],
             (4*k+3),total_loss[4*k+2,epoch],(4*k+4),total_loss[4*k+3,epoch]))

  if epoch >= opt.buffer:
      for i in range(opt.layer):
          for j in range(opt.ensemble):
              total_sparsity[j,epoch] += cal_sparsity(params[j][i].data)
      
      for k in range(opt.ensemble//4):
          print("Sparsity_%02d: %d, Sparsity_%02d: %d, Sparsity_%02d: %d, Sparsity_%02d: %d" %
                ((4*k+1),total_sparsity[4*k,epoch],(4*k+2),total_sparsity[4*k+1,epoch],
                 (4*k+3),total_sparsity[4*k+2,epoch],(4*k+4),total_sparsity[4*k+3,epoch]))
  
  if epoch >= opt.buffer:
                
      S = minmax_sparsity(total_sparsity[:,epoch].reshape(opt.ensemble))
      L = minmax_recip_loss(total_loss[:,epoch].reshape(opt.ensemble))
      prob = make_prob(opt.alpha*S,L,opt.ensemble)
      
      for k in range(opt.ensemble//4):
          print("Prob_%02d: %f, Prob_%02d: %f, Prob_%02d: %f, Prob_%02d: %f" % 
                ((4*k+1),prob[4*k], (4*k+2),prob[4*k+1], (4*k+3),prob[4*k+2], (4*k+4),prob[4*k+3]))
          
      for i in range(opt.layer):
          ## selection ##
          copies = []
          mask_list = []
          for j in range(opt.ensemble):
              copy = torch.empty((masks[j][i].shape))
              copies.append(copy)
              mask_list.append(masks[j][i])
      
          for j in range(opt.ensemble):
              #copy[j] = mask_list[torch.multinomial(prob,1)[0]].clone().detach()
              copies[j] = mask_list[np.random.choice(opt.ensemble, 1, p=prob.numpy())[0]].clone().detach()
          
          ## simple pairs : (copy_01, copy_02), (copy_03, copy_04) ##
          ## cross over ##
          for k in range(opt.ensemble//2):
              cross_over(copies[2*k],copies[2*k+1])
      
          ## mutation and inherit ##
          for j in range(opt.ensemble):
              Mutate(copies[j], opt.mutate, i, j)
              masks[j][i] = copies[j].clone().detach().float().cuda()
  
#### time ##########################################################################################################
stop = timeit.default_timer()
print("=" * 100)
print("걸린 시간은 %f 초" % (stop - start))

#### test ####
accuracy = np.empty((opt.ensemble))
with torch.no_grad():
    for j in range(opt.ensemble):
        models[j].eval()
        correct = 0
        for x, y in test_loader:
              
            x = Variable(x).float().cuda()
            y = Variable(y).long().cuda()
            
            output = models[j](x)
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
              
        accuracy[j] = correct / len(test_loader.dataset)

for k in range(opt.ensemble//4):
    print("Accuracy_%02d: %f, Accuracy_%02d: %f, Accuracy_%02d: %f, Accuracy_%02d: %f" % 
          ((4*k+1),accuracy[4*k], (4*k+2),accuracy[4*k+1], (4*k+3),accuracy[4*k+2], (4*k+4),accuracy[4*k+3]))

#### plot ##########################################################################################################
print("=" * 100)
Loss_plot_name = "Loss_%d.%d.%d.%d.png" % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
Loss_plot_path = os.path.join(main_path,Loss_plot_name)
for j in range(opt.ensemble):
    plt.plot(total_loss[j,:].numpy())
plt.axis([0,opt.n_epochs,0,2.5])
plt.savefig(Loss_plot_path)
plt.show()
plt.close()

Sparsity_plot_name = "Sparsity_%d.%d.%d.%d.png" % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
Sparsity_plot_path = os.path.join(main_path, Sparsity_plot_name)
for j in range(opt.ensemble):
    plt.plot(total_sparsity[j,:].numpy())
plt.axis([0,opt.n_epochs,0,cnt])
plt.savefig(Sparsity_plot_path)
plt.show()
plt.close()

print("\n")
print("=" * 100)
mean = []
for i in range(opt.layer):
    dist = np.empty((opt.ensemble))
    for j in range(opt.ensemble):
        dist[j] = 1-cal_sparsity(params[j][i].data).item()/params[j][i].data.reshape(-1,).shape[0]
    
    avg = dist.sum()/opt.ensemble
    mean.append(avg)
    print("Mean of sparsity rate in #%d layer: %f" % (i, avg))

Layer_plot_name = "Layer_%d.%d.%d.%d.png" % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
Layer_plot_path = os.path.join(main_path, Layer_plot_name)
y1_value = mean
x_name=('cv1', 'cv1.b', 'cv3', 'cv3.b', 'cv5','cv5.b', 'fc6', 'fc6.b', 'fc7', 'fc7.b')
n_groups = len(x_name)
index = np.arange(n_groups)

plt.bar(index, y1_value, tick_label=x_name, align='center')
plt.xlabel('layer')
plt.ylabel('Remaning Rate')
plt.title('Parameter Remaning Rate for each layer')
plt.xlim(-1, n_groups)
plt.ylim(0, 1)
plt.savefig(Layer_plot_path)
plt.show()
plt.close()
print("\n")

print("=" * 100)
print("<<<<<<<<<<Double Check Hyper Parameter>>>>>>>>>>")
print("Experiment Version              | %d.%d.%d.%d" % (opt.ver1, opt.ver2, opt.ver3, opt.ver4))
print("Total Epoch is                  | %d" % opt.n_epochs)
print("Batch size is                   | %d" % opt.batch_size)
print("Leraning rate is                | %f" % opt.lr)
print("Initial Mask rate is            | %f" % opt.mask)
print("Mutation probability is         | %f" % opt.mutate)
print("Balance factor alpha is         | %f" % opt.alpha)
print("How long buffer is              | %d" % opt.buffer)
print("Recovery constant is            | %f" % opt.recovery)
print("Recovery masking is             | %f" % opt.remask)
print("How many models are ensembled   | %d" % opt.ensemble)
print("How many layers will be pruned  | %d" % opt.layer)
if torch.cuda.is_available():
    print("Model will be build on          | %d GPU" % torch.cuda.device_count())
    for c in range(torch.cuda.device_count()):
        print("The current device is           | cuda:%d " % torch.cuda.current_device())
else:
    print("Model will be build on          | CPU")

print("=" * 100)

result_name = 'Result_%d.%d.%d.%d.tsv' % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
result_path = os.path.join(main_path, result_name)

f = open(result_path, 'w', encoding='utf-8', newline='')
wr = csv.writer(f, delimiter='\t')
wr.writerow("Sparsity")
for k in range(opt.ensemble//4):
    wr.writerow(total_sparsity[4*k:4*k+4,epoch].tolist())
wr.writerow("Loss")
for k in range(opt.ensemble//4):
    wr.writerow(total_loss[4*k:4*k+4,epoch].tolist())
wr.writerow("Accuracy")
for k in range(opt.ensemble//4):
	wr.writerow(accuracy[4*k:4*k+4].tolist())
wr.writerow("Layer")
wr.writerow(mean)
f.close()
print("Result is saved")
print("\n")

print("=" * 100)
Best_model_name = 'Best_%d.%d.%d.%d.pkl' % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
Best_model_path = os.path.join(main_path, Best_model_name)
torch.save({'epoch': epoch,
            'model_state_dict': models[prob.argmax()].state_dict(),
            'optimizer_state_dict': optims[prob.argmax()].state_dict(),
            'loss': total_loss[prob.argmax()]}, Best_model_path)
print("Best model #%d is saved" % (prob.argmax()+1))


    
















