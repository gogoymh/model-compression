import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import argparse
import timeit

#################################################################################################################
#from network import LeNet as Net
from network import resnet18 as Net

def cal_ch(params,changable,layer):
    total_start = timeit.default_timer()
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
                    print(i,k, "node")
                    limit = changable[idx+1] if (idx+1) != len(changable) else layer # opt.layer
                    for j in range(i+1,limit):
                        if params[j].data[k] == 0:
                            parallel_bonus += 1
                            print(j,k,'parallel')
            
        else: # Convolutional kenel channel pruning             
            for k in range(params[i].data.shape[0]): # output channel or node
                if params[i].data[k,0,0,0].item() != 0:
                    continue
                elif (params[i].data[k,:,:,:] != 0).sum().item() == 0:
                    channel_bonus += 1
                    print(i,k, "channel")
                    limit = changable[idx+1] if (idx+1) != len(changable) else layer # opt.layer
                    for j in range(i+1,limit):
                        if params[j].data[k] == 0:
                            parallel_bonus += 1
                            print(j,k,'parallel')
            for h in range(params[i].data.shape[2]):
                if (params[i].data[:,:,h,:] != 0).sum().item() == 0:
                    horizontal += 1
                    print(i,h,"horizontal")
            for v in range(params[i].data.shape[3]):
                if (params[i].data[:,:,:,v] != 0).sum().item() == 0:
                    vertical += 1
                    print(i,v,"vertical")
                
            rectangular_bonus = vertical + horizontal
    total_stop = timeit.default_timer()
    print('Time is',(total_stop-total_start))
    return channel_bonus, parallel_bonus, rectangular_bonus


def load_pretrain(pre_path, model):
    checkpoint = torch.load(pre_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    try:
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    except:
        epoch = None
        loss = None

    return model, optimizer, epoch, loss

class call_parameter(object):
    def __init__(self, model, ensemble):
        self.params = []
        parameter = list(model.parameters())
        for i in range(ensemble):
            self.params.append(parameter)
        
    def update(self, param_list):
        
        self.params = param_list
        
    def get_parameter(self):
        
        return self.params

#####################################################################################################################
train_loader = DataLoader(
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
                batch_size=128, shuffle=True, pin_memory = True)


criterion = nn.CrossEntropyLoss()

device = torch.device("cuda:0")
model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001)

model, optimizer, _, _ = load_pretrain("C:\유민형\개인 연구\model compression\Resnet18_Cifar10.pkl", model, optimizer)


####################################################################################################
parameter = call_parameter(model, 16) # opt.ensemble
params = parameter.get_parameter()


changable = []
for i in range(len(params[0])):
    if len(params[0][i].data.shape) != 1:
        changable.append(i)
print(changable)


#####################################################################################################################

#### Hyper parameter ################################################################################################
parser = argparse.ArgumentParser()
#parser.add_argument("--initcuda", type=int, default=0, help="Initial number of cuda")
#parser.add_argument("--numcuda", type=int, default=8, help="How many cuda to use")
parser.add_argument("--ver1", type=int, default=1, help="Version info")
parser.add_argument("--ver2", type=int, default=0, help="Version info")
parser.add_argument("--ver3", type=int, default=0, help="Version info")
parser.add_argument("--ver4", type=int, default=0, help="Version info")
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=5000, help="size of the batches")
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
print("Fine tuning                     | %d" % opt.fine)

#####################################################################################################################
class Main(object):
    def __init__(self):
        #### path ####
        self.main_path = '/home/super/ymh/modelcomp/results/%d.%d.%d.%d' % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
        #self.main_path = 'C:/유민형/개인 연구/model compression/results/%d.%d.%d.%d' % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
        print("Path is created                 |", self.main_path)
        
        if not os.path.isdir(self.main_path):
            os.mkdir(self.main_path)
        
        #### load_data ####
        self.train_loader = None
        self.test_loader = None
        
        #### build_network ####
        self.criterion = None
        self.model = None
        self.optimizer = None
        self.device = None
        
        #### create_mask ####
        self.masks = None
        
        #### load_pretrain ####
        self.params = None
        
        self.total_loss = None
        self.total_sparsity = None
        
        self.last_losses = None
        self.last_sparsities = None
        
        self.changable = None
        self.base_loss = None
        self.base_unstructured = None
        self.base_channel = None
        self.base_parallel = None
        self.base_rectangular = None
        
        #### genetic_algorithm ####
        self.best_model_tmp = None
        
        #### fine_tuning ####
        
        #### save_result ####
        
        
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
                                
        print("Length of train loader is       | %d" % len(self.train_loader)) # train sample 나누기 batch size
    
    def build_network(self):
        self.criterion = nn.CrossEntropyLoss()
        
        self.device = torch.device("cuda:0")
        self.model = Net(num_classes=10).to(self.device)

        #self.optimizer = optim.SGD(model.parameters(), lr=opt.lr)
        
    def create_mask(self):
        self.masks = []
        
        parameter = list(self.model.parameters())
        
        mask = []
        for i in range(opt.layer):
            mask_layer = torch.ones((parameter[i].data.shape))
            mask_layer = mask_layer.float().to(device)
            mask.append(mask_layer)
        
        for j in range(opt.ensemble):    
            self.masks.append(mask)
        
    def load_pretrain(self, pre_path, fisrt=True):
        checkpoint = torch.load(pre_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.params = []
        parameter = list(self.model.parameters())
        for j in range(opt.ensemble):
            self.params.append(parameter)
            
        self.total_loss = torch.zeros((opt.ensemble, opt.save))
        self.total_sparsity = torch.zeros((opt.ensemble, opt.save)).long()
        
        #### Base loss and sparsity to calculate objective function ####
        if first:
            self.base_loss = checkpoint['loss']            
            
            self.changable = []
            unstructured_cnt = 0
            output_ch = 0
            parallel_ch = 0
            rectangular_cnt = 0
            for i in range(opt.layer):
                unstructured_cnt += parameter[i].data.reshape(-1,).shape[0]
                if len(parameter[i].data.shape) != 1:
                    self.changable.append(i)
                    output_ch += parameter[i].data[0]
                    if len(parameter[i].data.shape) == 4:
                        rectangular_cnt += parameter[i].data[2] + parameter[i].data[3]
                elif len(parameter[i].data.shape) == 1:
                    parallel_ch += parameter[i].data[0]
            
            self.base_unstructured = unstructured_cnt
            print("How many total parameters       | %d * %d" % (unstructured_cnt, opt.ensemble))
            self.base_channel = output_ch
            self.base_parallel = parallel_ch
            self.base_rectangular = rectangular_cnt
            print("Layers for structured pruning   |", self.changable)
        
        #### Loss and Sparsity for genetic algorithm ####
        self.last_losses = torch.zeros((opt.ensemble))
        for j in range(opt.ensemble):
            self.last_losses[j] = checkpoint['loss']
            
        self.last_sparsities = torch.zeros((4, opt.ensemble))
        
        unstructured = 0
        for i in range(opt.layer):
            unstructured += cal_sparsity(parameter[i])
        
        channel_bonus, parallel_bonus, rectangular_bonus = cal_ch(parameter, self.changable, opt.layer)
        
        for j in range(opt.ensemble):
            self.last_sparsities[0,j] = unstructured
            self.last_sparsities[1,j] = channel_bonus
            self.last_sparsities[2,j] = parallel_bonus
            self.last_sparsities[3,j] = rectangular_bonus        
        
    def genetic_algorithm(self, epoch):
        L = self.base_loss/self.last_losses
        for k in range(opt.ensemble//4):
            print("Loss_%02d: %f, Loss_%02d: %f, Loss_%02d: %f, Loss_%02d: %f" % 
                  ((4*k+1),self.last_losses[4*k,epoch % opt.save],(4*k+2),self.last_losses[4*k+1,epoch % opt.save],
                   (4*k+3),self.last_losses[4*k+2,epoch % opt.save],(4*k+4),self.last_losses[4*k+3,epoch % opt.save]))
        
        self.last_sparsities[0,:] /= self.base_unstructured
        self.last_sparsities[1,:] /= self.base_channel
        self.last_sparsities[2,:] /= self.base_parallel
        self.last_sparsities[3,:] /= self.base_rectangular
        
        S = 0
        for k in range(4):
            S += self.last_sparsities[k,:]
        S /= 4
        for k in range(opt.ensemble//4):
            print("Sparsity_%02d: %d, Sparsity_%02d: %d, Sparsity_%02d: %d, Sparsity_%02d: %d" %
                  ((4*k+1),self.S[4*k],(4*k+2),self.S[4*k+1],
                   (4*k+3),self.S[4*k+2],(4*k+4),self.S[4*k+3]))
        
        prob = make_prob(opt.alpha*S,L,opt.ensemble)  
        self.best_model_tmp = prob.argmax()
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
                self.masks[j][i] = copies[j].clone().detach().float().to(self.device)

    def fine_tuning(self, epoch, fine):
        for step in range(fine):
            
            #### test if model is collapsed ####
            for j in range(opt.ensemble):
                if cal_sparsity(self.params[j][0].grad).item() == self.params[0][0].data.reshape(-1,).shape[0]:
                    print("!" * 30)
                    print("Model %d will be replaced with %d" %((j+1), (self.best_model_tmp+1)))
                    for i in range(opt.layer):
                        #self.params[j][i].data = self.params[self.best_model_tmp][i].data.clone().detach().to(self.device)
                        self.params[j][i].data = self.params[self.best_model_tmp][i]#.data.clone().detach().to(self.device)
                    print("!" * 30)
            
            #### remove weights ####
            for i in range(opt.layer):
                for j in range(opt.ensemble):
                    self.params[j][i].data *= self.masks[j][i]
            
            for batch_idx, (x, y) in enumerate(self.train_loader):
                
                x = x.float().to(self.device)
                y = y.long().to(self.device)
                
                parameter = list(self.model.parameters())
                for j in range(opt.ensemble):
                    
                    #### change weight ####
                    for i in range(opt.layer):
                        #parameter[i].data = self.params[j][i].data.clone().detach().to(self.device)
                        parameter[i].data = self.params[j][i].data#.clone().detach().to(self.device)
                        self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr)
                        
                    #### train ####
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    loss = self.criterion(output, y)
                    loss.backward()
                    
                    for i in range(opt.layer):
                        self.params[j][i].grad *= self.masks[j][i]
                    
                    self.optimizer.step()
                    self.total_loss[j, epoch % opt.save] += loss.item()
            
        for j in range(opt.ensemble):
            self.total_loss[j, epoch % opt.save] /= (len(self.train_loader) * fine)
            self.last_losses[j] = self.total_loss[j, epoch % opt.save]
        
            unstructured = 0
            for i in range(opt.layer):
                unstructured += cal_sparsity(params[j][i].data)
        
            channel_bonus, parallel_bonus, rectangular_bonus = cal_ch(params[j], self.changable, opt.layer)

            self.last_sparsities[0,j] = unstructured
            self.last_sparsities[1,j] = channel_bonus
            self.last_sparsities[2,j] = parallel_bonus
            self.last_sparsities[3,j] = rectangular_bonus
        
    #def save_result(self):

#####################################################################################################################
if __name__ == '__main__':
    
    obj = Main()
    
    obj.load_data()
    obj.build_network()
    obj.create_mask()
    obj.load_pretrain("C:\유민형\개인 연구\model compression\Resnet18_Cifar10.pkl", first=True)
    
    for epoch in range(opt.n_epochs):
        obj.genetic_algorithm(epoch)
        obj.fine_tuning(epoch, opt.fine)
        
        if (epoch+1) % opt.save == 0:
            #obj.save_result()
            obj.load_pretrain("path", first=False)



























