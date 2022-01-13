from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import argparse
import numpy as np
#import timeit

from network import resnet50 as ResNet
from Environment4 import Environment
from Agent5 import DDPG_Agent
from data_processor2 import processor
from utils2 import FIFO_plot_buffer

###########################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
parser.add_argument("--method", type=str, default="both", help="Method")
parser.add_argument("--init_delta1", type=float, default=1, help="Initial delta")
parser.add_argument("--init_delta2", type=float, default=0.25, help="Initial delta")
parser.add_argument("--delta_decay", type=float, default=0.997, help="Delta decay")
parser.add_argument("--lbound", type=float, default=0, help="Left action bound")
parser.add_argument("--rbound", type=float, default=1, help="Right action bound")
parser.add_argument("--discount", type=float, default=1, help="Discount factor")
parser.add_argument("--tau", type=float, default=0.01, help="Tau for soft update")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--memory_size", type=int, default=27000, help="Memory size")
parser.add_argument("--warmup", type=int, default=500, help="Warm up episodes")
parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate")
parser.add_argument("--episodes", type=int, default=4000, help="Total episodes")
parser.add_argument("--epsilon", type=float, default=0.01, help="Prevent edge case")
parser.add_argument("--beta", type=float, default=0.4, help="Important Sampling")
parser.add_argument("--beta_increment_per_sampling", type=float, default=0.001, help="Anneal beta")
parser.add_argument("--alpha", type=float, default=0.8, help="How much prioritization is used")
parser.add_argument("--flop", type=float, default=0.7, help="How much parameters are remained")
parser.add_argument("--acc", type=float, default=0.8, help="How much accuracy are required")
parser.add_argument("--pruned", type=float, default=0.1, help="How much flops are pruned")

opt = parser.parse_args()

###########################################################################################################################
'''
train_loader = DataLoader(
                datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=False,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=100, shuffle=False, pin_memory=True)
'''
                             
test_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
                        download=False,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=100, shuffle=False, pin_memory=True)

############################################################################################################################
env = Environment(ResNet, "C:/results/Resnet50_Cifar10.pth", test_loader, opt)

state, done = env.reset()

agent = DDPG_Agent(len(state), len(env.prunable_layer), opt)
data_controller = processor(len(state), len(env.prunable_layer), env.total_FLOPs, opt)

state, done = data_controller.get_embed_info(env) # reset is included
print(state)
print(done)
data_controller.state_reciever(np.array(list(state.values())))

plot_FIFO = FIFO_plot_buffer(opt.memory_size, len(env.prunable_layer))


env.original_result()

























