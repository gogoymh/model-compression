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
parser.add_argument("--numcuda", type=int, default=4, help="How many cuda to use")
parser.add_argument("--ver1", type=int, default=1, help="Version info")
parser.add_argument("--ver2", type=int, default=7, help="Version info")
parser.add_argument("--ver3", type=int, default=6, help="Version info")
parser.add_argument("--ver4", type=int, default=10, help="Version info")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
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

##################################################################################################################
class Main(object):
    def __init__(self):
        #self.main_path = '/home/super/ymh/modelcomp/results/%d.%d.%d.%d' % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
        self.main_path = 'C:/유민형/개인 연구/model compression/results/%d.%d.%d.%d' % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
        
        if not os.path.isdir(self.main_path):
            os.mkdir(self.main_path)
        
        self.train_loader = None
        self.test_loader = None
        
        self.criterion = None
        
        self.models = None
        self.optims = None
        self.params = None
        
        self.masks = None
        
        self.total_loss = None
        self.total_sparsity = None
        self.start = None
        
        self.cnt = None
        
        self.accuracy = None
        self.mean = None
        
    def load_data(self):
        self.train_loader = DataLoader(
                datasets.MNIST(
                        "../data/mnist",
                        train=True,
                        download=False,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.1307,), (0.3081,))]
                                ),
                        ),
                batch_size=opt.batch_size, shuffle=True, pin_memory = True)
                                
        self.test_loader = DataLoader(
                datasets.MNIST(
                        '../data/mnist',
                        train=False,
                        download=False,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.1307,), (0.3081,))]
                                ),
                        ),
                batch_size=1000, shuffle=False, pin_memory=True)
        print("Length of train loader is       | %d" % len(self.train_loader)) # train sample 나누기 batch size
    
    def build_network(self):
        #### Loss ####
        self.criterion = nn.CrossEntropyLoss()
        
        #### Models ####
        self.models = []
        self.optims = []
        self.params = []
        
        #### GPU checking ####
        if torch.cuda.is_available():
            print("Model will be build on          | %d GPU" % torch.cuda.device_count())
            for c in range(torch.cuda.device_count()):
                print("The current device is           | cuda:%d " % torch.cuda.current_device())
        else:
            print("Model will be build on          | CPU")
            
        gpu_list = list(range(opt.initcuda, opt.initcuda + opt.numcuda))
        print(gpu_list)
        ####
        for j in range(opt.ensemble):
            #torch.cuda.manual_seed(j)
            model = Net()
            if torch.cuda.device_count() > 1:
                #model = torch.nn.DataParallel(model, gpu_list).cuda()
                #cudnn.benchmark = True
                if j % 4 == 0:
                    device = torch.device("cuda:0")
                    model.to(device)
                elif j % 4 == 1:
                    device = torch.device("cuda:1")
                    model.to(device)
                elif j % 4 == 2:
                    device = torch.device("cuda:2")
                    model.to(device)
                else:
                    device = torch.device("cuda:3")
                    model.to(device)
            else:
                model.cuda()
            
            optimizer = optim.SGD(model.parameters(), lr=opt.lr)
            parameter = list(model.parameters())
            
            self.models.append(model)
            self.optims.append(optimizer)
            self.params.append(parameter)

    def count_parameters(self):
        self.cnt = 0
        for i in range(opt.layer):
            self.cnt += self.params[0][i].data.reshape(-1,).shape[0]
        print("How many total parameters       | %d" % self.cnt)

    def mask_initialize(self):
        self.masks = []
        
        for j in range(opt.ensemble):
            mask = []
            if j % 4 == 0:
                device = torch.device("cuda:0")
            elif j % 4 == 1:
                device = torch.device("cuda:1")
            elif j % 4 == 2:
                device = torch.device("cuda:2")
            else:
                device = torch.device("cuda:3")
            for i in range(opt.layer):
                #torch.manual_seed(j)
                mask_layer = torch.FloatTensor(self.params[j][i].data.shape).uniform_() > opt.mask
                mask_layer = mask_layer.float().to(device)
                mask.append(mask_layer)
            self.masks.append(mask)

    def train(self):
        self.total_loss = torch.zeros((opt.ensemble, opt.n_epochs))
        self.total_sparsity = torch.zeros((opt.ensemble, opt.n_epochs)).long()
        self.start = timeit.default_timer()
        
        for epoch in range(opt.n_epochs):
            print("=" * 100)
            print("Epoch: %d" % (epoch+1))
            
            if epoch >= opt.buffer:
                #### model broken test ####
                for j in range(opt.ensemble):
                    if j % 4 == 0:
                        device = torch.device("cuda:0")
                    elif j % 4 == 1:
                        device = torch.device("cuda:1")
                    elif j % 4 == 2:
                        device = torch.device("cuda:2")
                    elif j % 4 == 3:
                        device = torch.device("cuda:3")
                    elif cal_sparsity(self.params[j][0].grad).item() == self.params[0][0].data.reshape(-1,).shape[0]:
                        collapsed = 0
                        for i in range(2,9,2):
                            if cal_sparsity(self.params[j][i].grad).item() == self.params[0][i].data.reshape(-1,).shape[0] :
                                collapsed += 2
                        if collapsed >= 2:
                            self.params[j][collapsed-2].data += opt.recovery * (self.params[j][collapsed-2].data == 0).float() * (torch.FloatTensor(self.params[j][collapsed-2].data.shape).uniform_() > opt.remask).float().to(device)
                        self.params[j][collapsed].data += opt.recovery * (self.params[j][collapsed].data == 0).float() * (torch.FloatTensor(self.params[j][collapsed].data.shape).uniform_() > opt.remask).float().to(device)
                        print("!" * 30)
                        print("Layer %d of Model %d was collapsed. Get recovered!" % (collapsed, j+1))
                        print("!" * 30)
                #### weight removing ####
                for i in range(opt.layer):
                    for j in range(opt.ensemble):
                        self.params[j][i].data *= self.masks[j][i]
  
            #### train ####
            for batch_idx, (x, y) in enumerate(self.train_loader):
                
                x = Variable(x).float().cuda()
                y = Variable(y).long().cuda() 
                
                for j in range(opt.ensemble):                    
                    
                    self.optims[j].zero_grad()
                    if j % 4 == 0:
                        device = torch.device("cuda:0")
                    elif j % 4 == 1:
                        device = torch.device("cuda:1")
                    elif j % 4 == 2:
                        device = torch.device("cuda:2")
                    else:
                        device = torch.device("cuda:3")
                    output = self.models[j](x.to(device))        
                    loss = self.criterion(output, y.to(device))        
                    loss.backward()
                    
                    if epoch >= opt.buffer:
                        for i in range(opt.layer): # gradient removing
                            self.params[j][i].grad *= self.masks[j][i]
                    self.optims[j].step()        
                    self.total_loss[j,epoch] += loss.item()
            
            for j in range(opt.ensemble):
                self.total_loss[j,epoch] /= len(self.train_loader)
  
            #### print ####  
            for k in range(opt.ensemble//4):
                print("Loss_%02d: %f, Loss_%02d: %f, Loss_%02d: %f, Loss_%02d: %f" % 
                      ((4*k+1),self.total_loss[4*k,epoch],(4*k+2),self.total_loss[4*k+1,epoch],
                       (4*k+3),self.total_loss[4*k+2,epoch],(4*k+4),self.total_loss[4*k+3,epoch]))

            if epoch >= opt.buffer:
                for i in range(opt.layer):
                    for j in range(opt.ensemble):
                        self.total_sparsity[j,epoch] += cal_sparsity(self.params[j][i].data)
      
                for k in range(opt.ensemble//4):
                    print("Sparsity_%02d: %d, Sparsity_%02d: %d, Sparsity_%02d: %d, Sparsity_%02d: %d" %
                          ((4*k+1),self.total_sparsity[4*k,epoch],(4*k+2),self.total_sparsity[4*k+1,epoch],
                           (4*k+3),self.total_sparsity[4*k+2,epoch],(4*k+4),self.total_sparsity[4*k+3,epoch]))
  
            if epoch >= opt.buffer:
                
                S = minmax_sparsity(self.total_sparsity[:,epoch].reshape(opt.ensemble))
                print(S)
                L = minmax_recip_loss(self.total_loss[:,epoch].reshape(opt.ensemble))
                print(L)
                prob = make_prob(opt.alpha*S,L,opt.ensemble)
      
                for k in range(opt.ensemble//4):
                    print("Prob_%02d: %f, Prob_%02d: %f, Prob_%02d: %f, Prob_%02d: %f" % 
                          ((4*k+1),prob[4*k], (4*k+2),prob[4*k+1], (4*k+3),prob[4*k+2], (4*k+4),prob[4*k+3]))
          
                for i in range(opt.layer):
                    ## selection ##
                    copies = []
                    mask_list = []
                    for j in range(opt.ensemble):
                        copy = torch.empty((self.masks[j][i].shape))
                        copies.append(copy)
                        mask_list.append(self.masks[j][i])
      
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
                        if j % 4 == 0:
                            device = torch.device("cuda:0")
                        elif j % 4 == 1:
                            device = torch.device("cuda:1")
                        elif j % 4 == 2:
                            device = torch.device("cuda:2")
                        else:
                            device = torch.device("cuda:3")
                        self.masks[j][i] = copies[j].clone().detach().float().to(device)
        
        self.stop = timeit.default_timer()
        print("=" * 100)
        print("걸린 시간은 %f 초" % (self.stop - self.start))
        
        print("\n")
        
        print("=" * 100)
        Best_model_name = 'Best_%d.%d.%d.%d.pkl' % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
        Best_model_path = os.path.join(self.main_path, Best_model_name)
        torch.save({'epoch': epoch,
                    'model_state_dict': self.models[prob.argmax()].state_dict(),
                    'optimizer_state_dict': self.optims[prob.argmax()].state_dict(),
                    'loss': self.total_loss[prob.argmax()]}, Best_model_path)
        print("Best model #%d is saved" % (prob.argmax()+1))
        
    def test(self):
        self.accuracy = np.empty((opt.ensemble))
        
        with torch.no_grad():
            for j in range(opt.ensemble):
                if j % 4 == 0:
                    device = torch.device("cuda:0")
                elif j % 4 == 1:
                    device = torch.device("cuda:1")
                elif j % 4 == 2:
                    device = torch.device("cuda:2")
                else:
                    device = torch.device("cuda:3")                
                self.models[j].eval()
                correct = 0
                for x, y in self.test_loader:
                    
                    x = Variable(x).float().cuda()
                    y = Variable(y).long().cuda()
                    
                    output = self.models[j](x.to(device))
                    pred = output.argmax(1, keepdim=True)
                    correct += pred.eq(y.view_as(pred).to(device)).sum().item()
                    
                self.accuracy[j] = correct / len(self.test_loader.dataset)

        for k in range(opt.ensemble//4):
            print("Accuracy_%02d: %f, Accuracy_%02d: %f, Accuracy_%02d: %f, Accuracy_%02d: %f" % 
                  ((4*k+1),self.accuracy[4*k], (4*k+2),self.accuracy[4*k+1], (4*k+3),self.accuracy[4*k+2], (4*k+4),self.accuracy[4*k+3]))

    def plot(self):
        print("=" * 100)
        Loss_plot_name = "Loss_%d.%d.%d.%d.png" % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
        Loss_plot_path = os.path.join(self.main_path,Loss_plot_name)
        for j in range(opt.ensemble):
            plt.plot(self.total_loss[j,:].numpy())
        plt.axis([0,opt.n_epochs,0,2.5])
        plt.savefig(Loss_plot_path)
        plt.show()
        plt.close()
        
        Sparsity_plot_name = "Sparsity_%d.%d.%d.%d.png" % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
        Sparsity_plot_path = os.path.join(self.main_path, Sparsity_plot_name)
        for j in range(opt.ensemble):
            plt.plot(self.total_sparsity[j,:].numpy())
        plt.axis([0,opt.n_epochs,0, self.cnt])
        plt.savefig(Sparsity_plot_path)
        plt.show()
        plt.close()
        
        print("\n")
        print("=" * 100)
        self.mean = []
        for i in range(opt.layer):
            dist = np.empty((opt.ensemble))
            for j in range(opt.ensemble):
                dist[j] = 1-cal_sparsity(self.params[j][i].data).item()/self.params[j][i].data.reshape(-1,).shape[0]
    
            avg = dist.sum()/opt.ensemble
            self.mean.append(avg)
            print("Mean of sparsity rate in #%d layer: %f" % (i, avg))

        Layer_plot_name = "Layer_%d.%d.%d.%d.png" % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
        Layer_plot_path = os.path.join(self.main_path, Layer_plot_name)
        y1_value = self.mean
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

    def save(self):
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
        result_path = os.path.join(self.main_path, result_name)
        
        f = open(result_path, 'w', encoding='utf-8', newline='')
        wr = csv.writer(f, delimiter='\t')
        wr.writerow("Sparsity")
        for k in range(opt.ensemble//4):
            wr.writerow(self.total_sparsity[4*k:4*k+4,(opt.n_epochs-1)].tolist())
        wr.writerow("Loss")
        for k in range(opt.ensemble//4):
            wr.writerow(self.total_loss[4*k:4*k+4,(opt.n_epochs-1)].tolist())
        wr.writerow("Accuracy")
        for k in range(opt.ensemble//4):
            wr.writerow(self.accuracy[4*k:4*k+4].tolist())
        wr.writerow("Layer")
        wr.writerow(self.mean)
        f.close()
        print("Result is saved")
        
if __name__ == '__main__':
    
    obj = Main()
    
    obj.load_data()
    obj.build_network()
    obj.count_parameters()
    obj.mask_initialize()
    obj.train()
    obj.test()
    obj.plot()
    obj.save()





















