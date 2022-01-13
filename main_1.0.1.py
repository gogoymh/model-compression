from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import timeit

from network import resnet50 as ResNet
#from network2 import ResNet18 as ResNet
from Environment2 import Environment
from Agent3 import DDPG_Agent
from data_processor2 import processor
from utils2 import FIFO_plot_buffer

###########################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
parser.add_argument("--method", type=str, default="channel", help="Method")
parser.add_argument("--init_delta", type=float, default=0.25, help="Initial delta")
parser.add_argument("--delta_decay", type=float, default=0.99, help="Delta decay")
parser.add_argument("--lbound", type=float, default=0, help="Left action bound")
parser.add_argument("--rbound", type=float, default=1, help="Right action bound")
parser.add_argument("--discount", type=float, default=1, help="Discount factor")
parser.add_argument("--tau", type=float, default=0.01, help="Tau for soft update")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--memory_size", type=int, default=15000, help="Memory size")
parser.add_argument("--warmup", type=int, default=250, help="Warm up episodes")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--episodes", type=int, default=400, help="Total episodes")
parser.add_argument("--epsilon", type=float, default=0.01, help="Prevent edge case")
parser.add_argument("--beta", type=float, default=0.4, help="Important Sampling")
parser.add_argument("--beta_increment_per_sampling", type=float, default=0.001, help="Anneal beta")
parser.add_argument("--alpha", type=float, default=0.6, help="How much prioritization is used")

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
                batch_size=128, shuffle=True, pin_memory=True)
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

data_controller.state_reciever(np.array(list(state.values())))

plot_FIFO = FIFO_plot_buffer(opt.memory_size, len(env.prunable_layer))

## ---- Search ---- ##
for j in range(opt.episodes):
    start = timeit.default_timer()
    for i in range(len(env.prunable_layer)):
        action = agent.searching_action(j, np.array(list(state.values())), i, data_controller)
        state, done = env.step(action)
        
        data_controller.action_reciever(action)
        data_controller.move_idx()
        data_controller.state_reciever(np.array(list(state.values())))
        
        if done:
            x, y, reward = env.get_reward(state, 0.7, 0.6, 0.2, 1e-6)
            data_controller.reward_reciever(reward)
            current_states, actions, next_states, rewards, terminals = data_controller.convert_data() # returns numpy array
            
            agent.update_network(current_states, actions, next_states, rewards, terminals)
            
            plot_FIFO.add(x, y, reward)
            
            if agent.memory.n_entries > agent.warmup:
                print("="*100)
                print("[Episode: %d]\n[FLOP_ratio: %f, -Loss: %f, Reward: %f]\n[delta: %f] [Mean Reward Batch: %f]" % ((j+1), x, y, reward, agent.delta, rewards.mean().item()))
                
                before_x, before_y, before_z = plot_FIFO.get_past()
                
                current_x, current_y, current_z = plot_FIFO.get_current()
                
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(before_x, before_y, before_z, s=1, marker="o")
                ax.scatter(current_x, current_y, current_z, marker="^")
                
                ax.set_xlim(0.8,1)
                ax.set_ylim(-6,0)
                ax.set_zlim(-5,0)
                
                plt.show()
                plt.close()
                
            finish = timeit.default_timer()
            print("Single Episode: %f" % (finish - start))
            
            state, done = env.reset()
            data_controller.state_reciever(np.array(list(state.values())))
            
            start = timeit.default_timer()


state, done = env.reset()
## ---- Validate pruned model ---- ##
for i in range(len(env.prunable_layer)):
    print("="*100)
    print(state)
    action = agent.deterministic_action(np.array(list(state.values())))
    state, done = env.step(action)
    
    if done:
        print("="*100)
        print(state)
        x, y, reward = env.get_reward(state,0,0,0,0)
        print("="*100)
        print("<Validation>")
        #print("Pruned Model's FLOP_ratio is %f" % x)
        print("Pruned Model's -Loss is %f" % y)
        print("Pruned Model's Validation reward is %f" % reward)
        env.test(state,0.7, 0.6, 0.2, 1e-6)
        
env.original_result()















































