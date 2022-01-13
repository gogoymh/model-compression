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
import ast

#### Custom Library #################################################################################################
from functions import cal_sparsity, minmax_sparsity, minmax_recip_loss, make_prob, cross_over, Mutate
from network import resnet18 as Net

#### Hyper parameter ################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--initcuda", type=int, default=0, help="Initial number of cuda")
parser.add_argument("--numcuda", type=int, default=8, help="How many cuda to use")
parser.add_argument("--ver1", type=int, default=2, help="Version info")
parser.add_argument("--ver2", type=int, default=0, help="Version info")
parser.add_argument("--ver3", type=int, default=2, help="Version info")
parser.add_argument("--ver4", type=int, default=5, help="Version info")
parser.add_argument("--n_epochs", type=int, default=3, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=5000, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
#parser.add_argument("--mask", type=float, default=0.99, help="Masking rate for sparsity tensor")
parser.add_argument("--mutate", type=float, default=0.0016, help="Mutate probability")
parser.add_argument("--alpha", type=float, default=1, help="Ratio between loss and sparsity")
parser.add_argument("--buffer", type=int, default=1, help="Buffer for converge")
#parser.add_argument("--recovery", type=float, default=0.0001, help="Recovery constant")
#parser.add_argument("--remask", type=float, default=0.99, help="Recovery masking rate")
parser.add_argument("--ensemble", type=int, default=16, help="Number of models")
parser.add_argument("--layer", type=int, default=62, help="Number of layers")
parser.add_argument("--save", type=int, default=2, help="Number of saving")

opt = parser.parse_args()

print("=" * 100)
print("<<<<<<<<<<Double Check Hyper Parameter>>>>>>>>>>")
print("Experiment Version              | %d.%d.%d.%d" % (opt.ver1, opt.ver2, opt.ver3, opt.ver4))
print("Total Epoch is                  | %d" % opt.n_epochs)
print("Batch size is                   | %d" % opt.batch_size)
print("Leraning rate is                | %f" % opt.lr)
#print("Initial Mask rate is            | %f" % opt.mask)
print("Mutation probability is         | %f" % opt.mutate)
print("Balance factor alpha is         | %f" % opt.alpha)
print("How long buffer is              | %d" % opt.buffer)
#print("Recovery constant is            | %f" % opt.recovery)
#print("Recovery masking is             | %f" % opt.remask)
print("How many models are ensembled   | %d" % opt.ensemble)
print("How many layers will be pruned  | %d" % opt.layer)

##################################################################################################################
class Main(object):
    def __init__(self):
        self.main_path = '/home/super/ymh/modelcomp/results/%d.%d.%d.%d' % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
        #self.main_path = 'C:/유민형/개인 연구/model compression/results/%d.%d.%d.%d' % (opt.ver1, opt.ver2, opt.ver3, opt.ver4)
        print("Path is created                 |", self.main_path)
        
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
        
        self.multigpu = None
        
        self.argmax_prob = None
        
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
                batch_size=opt.batch_size, shuffle=True, pin_memory = True, num_workers = 10)

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
                batch_size=1000, shuffle=False, pin_memory=True, num_workers = 10)
                                
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
            
        ####
        if torch.cuda.device_count() > 1:
            self.multigpu = True
            print("Decision                        | Multiple GPU will be used")
            gpu_list = list(range(opt.initcuda, opt.initcuda + opt.numcuda))
            print("GPU list will be                |", gpu_list)
        else:
            print("Decision                        | Single GPU will be used")
        
        for j in range(opt.ensemble):
            #torch.cuda.manual_seed(j)
            model = Net()
            if torch.cuda.device_count() > 1:
                #model = torch.nn.DataParallel(model, gpu_list).cuda()
                #cudnn.benchmark = True
                device = torch.device("cuda:%d" % (j % opt.numcuda))
                model.to(device)

            else:
                device = torch.device("cuda:0")
                model.to(device)
            
            optimizer = optim.SGD(model.parameters(), lr=opt.lr)
            parameter = list(model.parameters())
            
            self.models.append(model)
            self.optims.append(optimizer)
            self.params.append(parameter)

    def count_parameters(self):
        self.cnt = 0
        for i in range(opt.layer):
            self.cnt += self.params[0][i].data.reshape(-1,).shape[0]
        print("How many total parameters       | %d * %d" % (self.cnt, opt.ensemble))

    def mask_initialize(self):
        self.masks = []
        
        for j in range(opt.ensemble):
            mask = []
            if self.multigpu:
                device = torch.device("cuda:%d" % (j % opt.numcuda))
            else:
                device = torch.device("cuda:0")

            for i in range(opt.layer):
                #torch.manual_seed(j)
                #mask_layer = torch.FloatTensor(self.params[j][i].data.shape).uniform_() > 0.99
                mask_layer = torch.ones((self.params[j][i].data.shape))
                mask_layer = mask_layer.float().to(device)
                mask.append(mask_layer)
            self.masks.append(mask)

    def train(self):
        self.total_loss = torch.zeros((opt.ensemble, opt.save))
        self.total_sparsity = torch.zeros((opt.ensemble, opt.save)).long()
        self.start = timeit.default_timer()
        
        if self.multigpu:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
            device2 = torch.device("cuda:2")
            device3 = torch.device("cuda:3")
            device4 = torch.device("cuda:4")
            device5 = torch.device("cuda:5")
            device6 = torch.device("cuda:6")
            device7 = torch.device("cuda:7")
        else:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
            device2 = torch.device("cuda:0")
            device3 = torch.device("cuda:0")
            device4 = torch.device("cuda:0")
            device5 = torch.device("cuda:0")
            device6 = torch.device("cuda:0")
            device7 = torch.device("cuda:0")
        
        for epoch in range(opt.n_epochs):
            print("=" * 100)
            print("Epoch: %d" % (epoch+1))
            epoch_start = timeit.default_timer()
            if epoch >= opt.buffer:
                #### model broken test ####
                for j in range(opt.ensemble):
                    if self.multigpu:
                        device = torch.device("cuda:%d" % (j % opt.numcuda))
                    else:
                        device = torch.device("cuda:0")

                    if cal_sparsity(self.params[j][0].grad).item() == self.params[0][0].data.reshape(-1,).shape[0]:
                        print("!" * 30)
                        print("Model %d will be replaced with %d" %((j+1), (self.argmax_prob+1)))
                        for i in range(opt.layer):
                            self.params[j][i].data = self.params[self.argmax_prob][i].data.clone().detach().to(device)
                        print("!" * 30)
                        
                #### weight removing ####
                for i in range(opt.layer):
                    for j in range(opt.ensemble):
                        self.params[j][i].data *= self.masks[j][i]
  
            #### train ####
            for batch_idx, (x, y) in enumerate(self.train_loader):
                
                x = Variable(x).float().cuda()
                y = Variable(y).long().cuda() 
                
                self.optims[0].zero_grad()
                self.optims[1].zero_grad()
                self.optims[2].zero_grad()
                self.optims[3].zero_grad()
                self.optims[4].zero_grad()
                self.optims[5].zero_grad()
                self.optims[6].zero_grad()
                self.optims[7].zero_grad()
                self.optims[8].zero_grad()
                self.optims[9].zero_grad()
                self.optims[10].zero_grad()
                self.optims[11].zero_grad()
                self.optims[12].zero_grad()
                self.optims[13].zero_grad()
                self.optims[14].zero_grad()
                self.optims[15].zero_grad()
                
                output0 = self.models[0](x.to(device0))
                output1 = self.models[1](x.to(device1))
                output2 = self.models[2](x.to(device2))
                output3 = self.models[3](x.to(device3))
                output4 = self.models[4](x.to(device4))
                output5 = self.models[5](x.to(device5))
                output6 = self.models[6](x.to(device6))
                output7 = self.models[7](x.to(device7))
                output8 = self.models[8](x.to(device0))
                output9 = self.models[9](x.to(device1))
                output10 = self.models[10](x.to(device2))
                output11 = self.models[11](x.to(device3))
                output12 = self.models[12](x.to(device4))
                output13 = self.models[13](x.to(device5))
                output14 = self.models[14](x.to(device6))
                output15 = self.models[15](x.to(device7))
                
                loss0 = self.criterion(output0, y.to(device0))
                loss1 = self.criterion(output1, y.to(device1))
                loss2 = self.criterion(output2, y.to(device2))
                loss3 = self.criterion(output3, y.to(device3))
                loss4 = self.criterion(output4, y.to(device4))
                loss5 = self.criterion(output5, y.to(device5))
                loss6 = self.criterion(output6, y.to(device6))
                loss7 = self.criterion(output7, y.to(device7))
                loss8 = self.criterion(output8, y.to(device0))
                loss9 = self.criterion(output9, y.to(device1))
                loss10 = self.criterion(output10, y.to(device2))
                loss11 = self.criterion(output11, y.to(device3))
                loss12 = self.criterion(output12, y.to(device4))
                loss13 = self.criterion(output13, y.to(device5))
                loss14 = self.criterion(output14, y.to(device6))
                loss15 = self.criterion(output15, y.to(device7))                          
                
                loss0.backward()
                loss1.backward()
                loss2.backward()
                loss3.backward()
                loss4.backward()
                loss5.backward()
                loss6.backward()
                loss7.backward()
                loss8.backward()
                loss9.backward()
                loss10.backward()
                loss11.backward()
                loss12.backward()
                loss13.backward()
                loss14.backward()
                loss15.backward()
                
                if epoch >= opt.buffer:
                    for i in range(opt.layer): # gradient removing
                        self.params[0][i].grad *= self.masks[0][i]
                        self.params[1][i].grad *= self.masks[1][i]
                        self.params[2][i].grad *= self.masks[2][i]
                        self.params[3][i].grad *= self.masks[3][i]
                        self.params[4][i].grad *= self.masks[4][i]
                        self.params[5][i].grad *= self.masks[5][i]
                        self.params[6][i].grad *= self.masks[6][i]
                        self.params[7][i].grad *= self.masks[7][i]
                        self.params[8][i].grad *= self.masks[8][i]
                        self.params[9][i].grad *= self.masks[9][i]
                        self.params[10][i].grad *= self.masks[10][i]
                        self.params[11][i].grad *= self.masks[11][i]
                        self.params[12][i].grad *= self.masks[12][i]
                        self.params[13][i].grad *= self.masks[13][i]
                        self.params[14][i].grad *= self.masks[14][i]
                        self.params[15][i].grad *= self.masks[15][i]
                        
                self.optims[0].step()
                self.optims[1].step()
                self.optims[2].step()
                self.optims[3].step()
                self.optims[4].step()
                self.optims[5].step()
                self.optims[6].step()
                self.optims[7].step()
                self.optims[8].step()
                self.optims[9].step()
                self.optims[10].step()
                self.optims[11].step()
                self.optims[12].step()
                self.optims[13].step()
                self.optims[14].step()
                self.optims[15].step()
                
                self.total_loss[0,epoch % opt.save] += loss0.item()
                self.total_loss[1,epoch % opt.save] += loss1.item()
                self.total_loss[2,epoch % opt.save] += loss2.item()
                self.total_loss[3,epoch % opt.save] += loss3.item()
                self.total_loss[4,epoch % opt.save] += loss4.item()
                self.total_loss[5,epoch % opt.save] += loss5.item()
                self.total_loss[6,epoch % opt.save] += loss6.item()
                self.total_loss[7,epoch % opt.save] += loss7.item()
                self.total_loss[8,epoch % opt.save] += loss8.item()
                self.total_loss[9,epoch % opt.save] += loss9.item()
                self.total_loss[10,epoch % opt.save] += loss10.item()
                self.total_loss[11,epoch % opt.save] += loss11.item()
                self.total_loss[12,epoch % opt.save] += loss12.item()
                self.total_loss[13,epoch % opt.save] += loss13.item()
                self.total_loss[14,epoch % opt.save] += loss14.item()
                self.total_loss[15,epoch % opt.save] += loss15.item()
                
            for j in range(opt.ensemble):
                self.total_loss[j,epoch % opt.save] /= len(self.train_loader)
            epoch_stop = timeit.default_timer()
            #### print ####  
            for k in range(opt.ensemble//4):
                print("Loss_%02d: %f, Loss_%02d: %f, Loss_%02d: %f, Loss_%02d: %f" % 
                      ((4*k+1),self.total_loss[4*k,epoch % opt.save],(4*k+2),self.total_loss[4*k+1,epoch % opt.save],
                       (4*k+3),self.total_loss[4*k+2,epoch % opt.save],(4*k+4),self.total_loss[4*k+3,epoch % opt.save]))
            print("Time for one epoch is", (epoch_stop-epoch_start))
            if epoch >= opt.buffer:
                for i in range(opt.layer):
                    for j in range(opt.ensemble):
                        self.total_sparsity[j,epoch % opt.save] += cal_sparsity(self.params[j][i].data)
      
                for k in range(opt.ensemble//4):
                    print("Sparsity_%02d: %d, Sparsity_%02d: %d, Sparsity_%02d: %d, Sparsity_%02d: %d" %
                          ((4*k+1),self.total_sparsity[4*k,epoch % opt.save],(4*k+2),self.total_sparsity[4*k+1,epoch % opt.save],
                           (4*k+3),self.total_sparsity[4*k+2,epoch % opt.save],(4*k+4),self.total_sparsity[4*k+3,epoch % opt.save]))
            
            #### Genetic Algorithm ####
            if epoch >= opt.buffer:
                S = minmax_sparsity(self.total_sparsity[:,epoch % opt.save].reshape(opt.ensemble))
                L = minmax_recip_loss(self.total_loss[:,epoch % opt.save].reshape(opt.ensemble))
                prob = make_prob(opt.alpha*S,L,opt.ensemble)
                self.argmax_prob = prob.argmax()
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
                        if self.multigpu:
                            device = torch.device("cuda:%d" % (j % opt.numcuda))
                        else:
                            device = torch.device("cuda:0")

                        self.masks[j][i] = copies[j].clone().detach().float().to(device)
                        
            #### record ####
            if (epoch+1) % opt.save == 0:
                #### test ####
                print("="*100)
                self.accuracy = np.empty((opt.ensemble))
        
                with torch.no_grad():
                    for j in range(opt.ensemble):
                
                        if self.multigpu:
                            device = torch.device("cuda:%d" % (j % opt.numcuda))
                        else:
                            device = torch.device("cuda:0")

                        self.models[j].eval()
                        correct = 0
                        for x, y in self.test_loader:
                    
                            x = Variable(x).float()
                            y = Variable(y).long()
                    
                        output = self.models[j](x.to(device))
                        pred = output.argmax(1, keepdim=True)
                        correct += pred.eq(y.to(device).view_as(pred)).sum().item()
                    
                        self.accuracy[j] = correct / len(self.test_loader.dataset)

                for k in range(opt.ensemble//4):
                    print("Accuracy_%02d: %f, Accuracy_%02d: %f, Accuracy_%02d: %f, Accuracy_%02d: %f" % 
                          ((4*k+1),self.accuracy[4*k], (4*k+2),self.accuracy[4*k+1], (4*k+3),self.accuracy[4*k+2], (4*k+4),self.accuracy[4*k+3]))
                
                #### model save ####
                Best_model_name = '%d_Best_%d.%d.%d.%d.pkl' % (opt.save*((epoch+1) // opt.save), opt.ver1, opt.ver2, opt.ver3, opt.ver4)
                Best_model_path = os.path.join(self.main_path, Best_model_name)
                torch.save({'epoch': epoch,
                            'model_state_dict': self.models[self.argmax_prob].state_dict(),
                            'optimizer_state_dict': self.optims[self.argmax_prob].state_dict(),
                            'loss': self.total_loss[self.argmax_prob]}, Best_model_path)
                print("\n")
                print("Best model #%d is saved in" % (self.argmax_prob+1), self.main_path)
                      
                
                #### plot ####
                print("=" * 100)
                
                ## loss ##
                Loss_plot_name = "%d_Loss_%d.%d.%d.%d.png" % (opt.save*((epoch+1) // opt.save), opt.ver1, opt.ver2, opt.ver3, opt.ver4)
                Loss_plot_path = os.path.join(self.main_path,Loss_plot_name)
                                
                if ((epoch+1) // opt.save) >= 2:
                    
                    sparsity_tmp = np.zeros((opt.ensemble,opt.save))
                    loss_tmp = np.zeros((opt.ensemble,opt.save))
                    
                    sparsity_tmp2 = np.zeros((opt.ensemble,opt.save))
                    loss_tmp2 = np.zeros((opt.ensemble,opt.save))
                    
                    for h in range(((epoch+1) // opt.save)-1):
                        results_before = '%d_Result_%d.%d.%d.%d.txt' % (opt.save*(h+1), opt.ver1, opt.ver2, opt.ver3, opt.ver4)
                        results_before_path = os.path.join(self.main_path,results_before)
                    
                        f = open(results_before_path,'r', encoding='utf-8', newline='')
                        lines = csv.reader(f, delimiter='\t')
                        a = [line for line in lines]
                        
                        for j in range(opt.ensemble):
                            sparsity_tmp[j] = np.array(ast.literal_eval(a[1][j]))
                            loss_tmp[j] = np.array(ast.literal_eval(a[3][j]))
                        
                        if h > 0:
                            sparsity_tmp2 = np.concatenate((sparsity_tmp2, sparsity_tmp), axis=1)
                            loss_tmp2 = np.concatenate((loss_tmp2, loss_tmp), axis=1)
                        else:
                            sparsity_tmp2 = sparsity_tmp.copy()
                            loss_tmp2 = loss_tmp.copy()
                    
                    sparsity_plot = np.concatenate((sparsity_tmp2, self.total_sparsity.numpy()), axis=1)
                    loss_plot = np.concatenate((loss_tmp2, self.total_loss.numpy()), axis=1)
                
                else:
                    sparsity_plot = self.total_sparsity.numpy()
                    loss_plot = self.total_loss.numpy()
                    
                for j in range(opt.ensemble):
                    plt.plot(loss_plot[j,:])
                plt.axis([0,epoch,0,2.5])
                plt.savefig(Loss_plot_path)
                plt.show()
                plt.close()
                
                ## sparsity ##
                Sparsity_plot_name = "%d_Sparsity_%d.%d.%d.%d.png" % (opt.save*((epoch+1) // opt.save), opt.ver1, opt.ver2, opt.ver3, opt.ver4)
                Sparsity_plot_path = os.path.join(self.main_path, Sparsity_plot_name)
                for j in range(opt.ensemble):
                    plt.plot(sparsity_plot[j,:])
                plt.axis([0,epoch,0, self.cnt])
                plt.savefig(Sparsity_plot_path)
                plt.show()
                plt.close()
                
                print("=" * 100)
                ## layer sparsity ##
                self.mean = []
                for i in range(opt.layer):
                    dist = np.empty((opt.ensemble))
                    for j in range(opt.ensemble):
                        dist[j] = 1-cal_sparsity(self.params[j][i].data).item()/self.params[j][i].data.reshape(-1,).shape[0]
    
                    avg = dist.sum()/opt.ensemble
                    self.mean.append(avg)
                    print("Mean of sparsity rate in #%d layer: %f" % (i, avg))

                Layer_plot_name = "%d_Layer_%d.%d.%d.%d.png" % (opt.save*((epoch+1) // opt.save), opt.ver1, opt.ver2, opt.ver3, opt.ver4)
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
                print("=" * 100)
                print("Plots are saved in", self.main_path)
                
                #### loss and sparsity record ####
                result_name = '%d_Result_%d.%d.%d.%d.txt' % (opt.save*((epoch+1) // opt.save), opt.ver1, opt.ver2, opt.ver3, opt.ver4)
                result_path = os.path.join(self.main_path, result_name)
                
                f = open(result_path, 'w', encoding='utf-8', newline='')
                wr = csv.writer(f, delimiter='\t')
                wr.writerow("Sparsity")
                wr.writerow(self.total_sparsity.tolist())
                wr.writerow("Loss")
                wr.writerow(self.total_loss.tolist())
                wr.writerow("Accuracy")
                wr.writerow(self.accuracy.tolist())
                wr.writerow("Layer")
                wr.writerow(self.mean)
                f.close()
                print("Result is recorded in", self.main_path)
                
                self.total_loss = torch.zeros((opt.ensemble, opt.save))
                self.total_sparsity = torch.zeros((opt.ensemble, opt.save)).long()
                
        self.stop = timeit.default_timer()
        print("=" * 100)
        print("걸린 시간은 %f 초" % (self.stop - self.start))
        
        print("\n")
                
if __name__ == '__main__':
    
    obj = Main()
    
    obj.load_data()
    obj.build_network()
    obj.count_parameters()
    obj.mask_initialize()
    obj.train()






















