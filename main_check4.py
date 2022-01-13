from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import timeit

from network import LeNet
#from network2 import ResNet18 as ResNet
from Environment4 import Environment
from Agent5 import DDPG_Agent
from data_processor2 import processor
from utils2 import FIFO_plot_buffer

###########################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
parser.add_argument("--method", type=str, default="rectangular", help="Method")
parser.add_argument("--init_delta1", type=float, default=1, help="Initial delta")
parser.add_argument("--init_delta2", type=float, default=0.25, help="Initial delta")
parser.add_argument("--delta_decay", type=float, default=0.999, help="Delta decay")
parser.add_argument("--lbound", type=float, default=0, help="Left action bound")
parser.add_argument("--rbound", type=float, default=1, help="Right action bound")
parser.add_argument("--discount", type=float, default=1, help="Discount factor")
parser.add_argument("--tau", type=float, default=0.01, help="Tau for soft update")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--memory_size", type=int, default=5000, help="Memory size")
parser.add_argument("--warmup", type=int, default=500, help="Warm up episodes")
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--episodes", type=int, default=2000, help="Total episodes")
parser.add_argument("--epsilon", type=float, default=0.01, help="Prevent edge case")
parser.add_argument("--beta", type=float, default=0.4, help="Important Sampling")
parser.add_argument("--beta_increment_per_sampling", type=float, default=0.001, help="Anneal beta")
parser.add_argument("--alpha", type=float, default=0.8, help="How much prioritization is used")
parser.add_argument("--flop", type=float, default=0.7, help="How much parameters are remained")
parser.add_argument("--acc", type=float, default=0.9, help="How much accuracy are required")
parser.add_argument("--pruned", type=float, default=0.5, help="How much flops are pruned")

opt = parser.parse_args()

###########################################################################################################################
'''
train_loader = DataLoader(
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
                batch_size=128, shuffle=True, pin_memory = True)
'''
                             
test_loader = DataLoader(
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
                batch_size=100, shuffle=False, pin_memory=True)

############################################################################################################################
env = Environment(LeNet, "C:/results/Lenet5_Mnist.pkl", test_loader, opt)

state, done = env.reset()

agent = DDPG_Agent(len(state), len(env.prunable_layer), opt)
data_controller = processor(len(state), len(env.prunable_layer), env.total_FLOPs, opt)

state, done = data_controller.get_embed_info(env) # reset is included

data_controller.state_reciever(np.array(list(state.values())))

i = 1
action = np.array([0.44694486, 0.9998864])
state, done = env.step(action)

i = 2
action = np.array([0.6436129, 0.99979514])
state, done = env.step(action)

i = 3
action = np.array([0.97305536, 0.99801123])
state, done = env.step(action)

i = 4
action = np.array([0.9581917, 0.9718188])
state, done = env.step(action)

i = 5
action = np.array([0.9674928, 0.96260524])
state, done = env.step(action)

x, y, reward = env.get_reward(state)
print("="*100)
print("<Validation>")
print("Pruned Model's Accuracy is %f" % y)
print("Pruned Model's Pruned Ratio is %f" % x)
print("Pruned Model's Validation reward is %f" % reward)

env.test(state)



