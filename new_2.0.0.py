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
#from network import LeNet as Net
from network import resnet18 as Net
from utils import cal_sparsity, cal_ch, make_prob, cross_over, Mutate#, minmax_fn

#### Hyper parameter ################################################################################################
parser = argparse.ArgumentParser()
#parser.add_argument("--initcuda", type=int, default=0, help="Initial number of cuda")
parser.add_argument("--numcuda", type=int, default=6, help="How many cuda to use")
parser.add_argument("--ver1", type=int, default=2, help="Version info")
parser.add_argument("--ver2", type=int, default=0, help="Version info")
parser.add_argument("--ver3", type=int, default=0, help="Version info")
parser.add_argument("--ver4", type=int, default=0, help="Version info")
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.1, help="SGD learning rate")
parser.add_argument("--mutate", type=float, default=0.0016, help="Mutate probability")
parser.add_argument("--alpha", type=float, default=0.5, help="Ratio between loss and sparsity")
#parser.add_argument("--buffer", type=int, default=100, help="Buffer for converge")
parser.add_argument("--ensemble", type=int, default=16, help="Number of models")
parser.add_argument("--layer", type=int, default=62, help="Number of layers")
parser.add_argument("--save", type=int, default=500, help="Number of saving")
parser.add_argument("--fine", type=int, default=1, help="Fine tuning time")

opt = parser.parse_args()

print("=" * 100)
print("<<<<<<<<<<Double Check Hyper Parameter>>>>>>>>>>")
print("Experiment Version              | %d.%d.%d.%d" % (opt.ver1, opt.ver2, opt.ver3, opt.ver4))
print("Total Epoch is                  | %d" % opt.n_epochs)
print("Batch size is                   | %d" % opt.batch_size)
print("Leraning rate is                | %f" % opt.lr)
print("Mutation probability is         | %f" % opt.mutate)
print("Balance factor alpha is         | %f" % opt.alpha)
#print("How long buffer is              | %d" % opt.buffer)
print("How many models are ensembled   | %d" % opt.ensemble)
print("How many layers will be pruned  | %d" % opt.layer)
print("Model will be saved by          | %d" % opt.save)
print("Fine tuning                     | %d" % opt.fine)

#####################################################################################################################
class Main(object):
    def __init__(self):
        #### path ####
        self.main_path = '/home/super/ymh/modelcomp/results/new_%d.%d.%d.%d' % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
        #self.main_path = 'C:/?????????/?????? ??????/model compression/results/new_%d.%d.%d.%d' % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
        print("Path is created                 |", self.main_path)
        
        if not os.path.isdir(self.main_path):
            os.mkdir(self.main_path)
        
        #### load_data ####
        self.train_loader = None
        self.test_loader = None
        
        #### build_network ####
        self.criterion = None
        
        self.models = None
        self.optims = None
        self.params = None
        
        #### create_mask ####
        self.masks = None
        
        #### load_pretrain ####        
        self.total_loss = None
        self.total_sparsity = None
        
        self.last_accuracies = None
        self.last_sparsities = None
        
        self.changable = None
        self.base_loss = None
        self.base_unstructured = None
        self.base_channel = None
        self.base_parallel = None
        self.base_rectangular = None
    
        self.prob = None
        
        #### genetic_algorithm ####
        self.best_model_tmp = None
        
        #### fine_tuning ####
        
        #### save_result ####
        self.model_save_path = None
        
    def load_data(self):
        self.train_loader = DataLoader(
                datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=False,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=opt.batch_size, shuffle=True, pin_memory = True, num_workers = 5)

        self.test_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
                        download=False,
                        transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=1000, shuffle=False, pin_memory=True, num_workers = 5)
                                
        print("Length of train loader is       | %d" % len(self.train_loader)) # train sample ????????? batch size
        print("Length of test loader is        | %d" % len(self.test_loader))
    def build_network(self):
        self.criterion = nn.CrossEntropyLoss()
        
        self.models = []
        self.optims = []
        self.params = []
        
        for j in range(opt.ensemble):
            model = Net()
            
            if torch.cuda.device_count() > 1:
                device = torch.device("cuda:%d" % (j % opt.numcuda))
            else:
                device = torch.device("cuda:0")
            
            model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
            parameter = list(model.parameters())
            
            self.models.append(model)
            self.optims.append(optimizer)
            self.params.append(parameter)
        
    def create_mask(self):
        self.masks = []
        
        for j in range(opt.ensemble):
            mask = []
            
            if torch.cuda.device_count() > 1:
                device = torch.device("cuda:%d" % (j % opt.numcuda))
            else:
                device = torch.device("cuda:0")
            
            for i in range(opt.layer):
                mask_layer = torch.ones((self.params[j][i].data.shape))
                mask_layer = mask_layer.float().to(device)
                mask.append(mask_layer)            
            
            self.masks.append(mask)
        
    def load_pretrain(self, pre_path=None, first=True):
        #### check path ####
        if first:
            checkpoint = torch.load(pre_path)
            print(pre_path,"is loaded")
        else:
            checkpoint = torch.load(self.model_save_path)
            print(self.model_save_path,"is loaded")
        
        #### load model ####
        self.optims = []
        self.params = []
        for j in range(opt.ensemble):
            self.models[j].load_state_dict(checkpoint['model_state_dict'])
            
            optimizer = optim.SGD(self.models[j].parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
            parameter = list(self.models[j].parameters())

            self.optims.append(optimizer)
            self.params.append(parameter)
        
        #### create recording tensor ####
        self.total_loss = torch.zeros((opt.ensemble, opt.save))
        self.total_sparsity = torch.zeros((opt.ensemble, opt.save))
        
        #### Base loss and sparsity to calculate objective function ####
        if first:
            self.changable = []
            unstructured_cnt = 0
            output_ch = 0
            parallel_ch = 0
            rectangular_cnt = 0
            for i in range(opt.layer):
                unstructured_cnt += self.params[0][i].data.reshape(-1,).shape[0]
                if len(self.params[0][i].data.shape) != 1:
                    self.changable.append(i)
                    output_ch += self.params[0][i].data.shape[0]
                    if len(self.params[0][i].data.shape) == 4:
                        rectangular_cnt += self.params[0][i].data.shape[2] + self.params[0][i].data.shape[3]
                elif len(self.params[0][i].data.shape) == 1:
                    parallel_ch += self.params[0][i].data.shape[0]
            
            self.base_unstructured = unstructured_cnt
            print("How many total parameters       | %d * %d" % (unstructured_cnt, opt.ensemble))
            self.base_channel = output_ch
            self.base_parallel = parallel_ch
            self.base_rectangular = rectangular_cnt
            print("Layers for structured pruning   |", self.changable)
            
    def evaluate(self):
        ev_start = timeit.default_timer()
        #### Accuracy ####
        self.last_accuracies = torch.zeros((opt.ensemble))            
        
        correct = torch.zeros((opt.ensemble))
        for x, y in self.test_loader:
            for j in range(opt.ensemble):
                if torch.cuda.device_count() > 1:
                    device = torch.device("cuda:%d" % (j % opt.numcuda))
                else:
                    device = torch.device("cuda:0")
                output = self.models[j](x.float().to(device))
                pred = output.argmax(1, keepdim=True)
                correct[j] += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
        self.last_accuracies = correct / len(self.test_loader.dataset)
        
        #### Sparsity ####
        self.last_sparsities = torch.zeros((4, opt.ensemble)).float()
        
        for j in range(opt.ensemble):
            unstructured = 0
            for i in range(opt.layer):
                unstructured += cal_sparsity(self.params[j][i])
        
            channel_bonus, parallel_bonus, rectangular_bonus = cal_ch(self.params[j], self.changable, opt.layer)

            self.last_sparsities[0,j] = unstructured
            self.last_sparsities[1,j] = channel_bonus
            self.last_sparsities[2,j] = parallel_bonus
            self.last_sparsities[3,j] = rectangular_bonus
            
        #### objective values ####
        print("="*100)
        print("[Evaluate Results]")
        for k in range(opt.ensemble//4):
            print("Accuracy_%02d: %f, Accuracy_%02d: %f, Accuracy_%02d: %f, Accuracy_%02d: %f" % 
                  ((4*k+1),self.last_accuracies[4*k],(4*k+2),self.last_accuracies[4*k+1],
                   (4*k+3),self.last_accuracies[4*k+2],(4*k+4),self.last_accuracies[4*k+3]))
        
        print(self.last_sparsities)
        self.last_sparsities[0,:] /= self.base_unstructured
        self.last_sparsities[1,:] /= self.base_channel
        self.last_sparsities[2,:] /= self.base_parallel
        self.last_sparsities[3,:] /= self.base_rectangular
        
        S = 0
        for k in range(4):
            S += self.last_sparsities[k,:]
        S /= 4
        
        for k in range(opt.ensemble//4):
            print("Sparsity_%02d: %f, Sparsity_%02d: %f, Sparsity_%02d: %f, Sparsity_%02d: %f" %
                  ((4*k+1),S[4*k],(4*k+2),S[4*k+1],
                   (4*k+3),S[4*k+2],(4*k+4),S[4*k+3]))
        
        self.prob = make_prob(opt.alpha*S,self.last_accuracies,opt.ensemble)  
        self.best_model_tmp = self.prob.argmax()
        for k in range(opt.ensemble//4):
            print("Prob_%02d: %f, Prob_%02d: %f, Prob_%02d: %f, Prob_%02d: %f" %
                  ((4*k+1),self.prob[4*k], (4*k+2),self.prob[4*k+1], (4*k+3),self.prob[4*k+2], (4*k+4),self.prob[4*k+3]))
        
        ev_stop = timeit.default_timer()
        print("Evaluation time is",(ev_stop-ev_start))
        
    def genetic_algorithm(self, epoch):
        print("="*100)
        ga_start = timeit.default_timer()
        
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
                copies[j] = mask_list[np.random.choice(opt.ensemble, 1, p=self.prob.numpy())[0]].clone().detach()
            
            ## simple pairs : (copy_01, copy_02), (copy_03, copy_04) ##
            ## cross over ##
            for k in range(opt.ensemble//2):
                cross_over(copies[2*k],copies[2*k+1])
      
            ## mutation and inherit ##
            for j in range(opt.ensemble):
                if torch.cuda.device_count() > 1:
                    device = torch.device("cuda:%d" % (j % opt.numcuda))
                else:
                    device = torch.device("cuda:0")
                Mutate(copies[j], opt.mutate, i, j)
                self.masks[j][i] = copies[j].clone().detach().float().to(device)
        
        ga_stop = timeit.default_timer()
        print("Genetic Algorithm time is",(ga_stop-ga_start))
        
    def fine_tuning(self, epoch, fine):
        print("="*100)
        ft0_start = timeit.default_timer()
        
        for step in range(fine):
            #### remove weight ####
            for j in range(opt.ensemble):
                for i in range(opt.layer):
                    self.params[j][i].data *= self.masks[j][i]
                
            for x, y in self.train_loader:   
                
                for j in range(opt.ensemble):
                    if torch.cuda.device_count() > 1:
                        device = torch.device("cuda:%d" % (j % opt.numcuda))
                    else:
                        device = torch.device("cuda:0")
                    #### train ####                    
                    self.optims[j].zero_grad()
                    output = self.models[j](x.float().to(device))
                    loss = self.criterion(output, y.long().to(device))
                    loss.backward()
                
                    for i in range(opt.layer):
                        self.params[j][i].grad *= self.masks[j][i]
                
                    self.optims[j].step()
                    
                    #### test if model is collapsed ####
                    if cal_sparsity(self.params[j][0].grad).item() == self.params[j][0].data.reshape(-1,).shape[0]:
                        print("!" * 30)
                        print("Model %d will be replaced with %d" %((j+1), (self.best_model_tmp+1)))
                        for i in range(opt.layer):
                            self.params[j][i].data = self.params[self.best_model_tmp][i].data.clone().to(device)
                        print("!" * 30)
        
        ft0_stop = timeit.default_timer()
        print("Fine tuning time is",(ft0_stop-ft0_start))
        
    def save_result(self, epoch):
        self.model_save_path = os.path.join(self.main_path, "result_%d.pkl" % (epoch+1))
        
        #### choose best model ####
        self.parameter = self.params[self.best_model_tmp].copy()
        self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr)
        
        #### copy best mask ####
        for j in range(opt.ensemble):
            self.masks[j] = self.masks[self.best_model_tmp].copy()
        
        #### save model ####
        print("Best model %d will be saved" % (self.best_model_tmp+1))
        torch.save({'epoch': epoch,
                    'model_state_dict': self.models[self.best_model_tmp].state_dict(),
                    'optimizer_state_dict': self.optims[self.best_model_tmp].state_dict()}, self.model_save_path)

#####################################################################################################################
if __name__ == '__main__':
    
    obj = Main()
    
    obj.load_data()
    obj.build_network()
    obj.create_mask()
    #obj.load_pretrain("C:\?????????\?????? ??????\model compression\Resnet18_Cifar10.pkl", first=True)
    obj.load_pretrain("/home/super/ymh/modelcomp/results/Resnet18_Cifar10.pkl", first=True)
    obj.evaluate()
    
    for epoch in range(opt.n_epochs):
        print("="*100)
        print("[Epoch:%d]" % (epoch+1))    
        obj.genetic_algorithm(epoch)
        obj.fine_tuning(epoch, opt.fine)
        obj.evaluate()
        
        if (epoch+1) % opt.save == 0:
            obj.save_result(epoch)
            obj.load_pretrain(first=False)
            obj.evaluate()



























