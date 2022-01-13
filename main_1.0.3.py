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
parser.add_argument("--method", type=str, default="both", help="Method")
parser.add_argument("--init_delta1", type=float, default=1, help="Initial delta")
parser.add_argument("--init_delta2", type=float, default=0.25, help="Initial delta")
parser.add_argument("--delta_decay", type=float, default=0.997, help="Delta decay")
parser.add_argument("--lbound", type=float, default=0, help="Left action bound")
parser.add_argument("--rbound", type=float, default=1, help="Right action bound")
parser.add_argument("--discount", type=float, default=1, help="Discount factor")
parser.add_argument("--tau", type=float, default=0.01, help="Tau for soft update")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--memory_size", type=int, default=5000, help="Memory size")
parser.add_argument("--warmup", type=int, default=500, help="Warm up episodes")
parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate")
parser.add_argument("--episodes", type=int, default=4000, help="Total episodes")
parser.add_argument("--epsilon", type=float, default=0.01, help="Prevent edge case")
parser.add_argument("--beta", type=float, default=0.4, help="Important Sampling")
parser.add_argument("--beta_increment_per_sampling", type=float, default=0.001, help="Anneal beta")
parser.add_argument("--alpha", type=float, default=0.8, help="How much prioritization is used")
parser.add_argument("--flop", type=float, default=0.7, help="How much parameters are remained")
parser.add_argument("--acc", type=float, default=1, help="How much accuracy are required")
parser.add_argument("--pruned", type=float, default=1, help="How much flops are pruned")

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

plot_FIFO = FIFO_plot_buffer(opt.memory_size, len(env.prunable_layer))

## ---- Search ---- ##
for j in range(opt.episodes):
    start = timeit.default_timer()
    for i in range(len(env.prunable_layer)):
        action = agent.searching_action3(j, np.array(list(state.values())), i, data_controller)
        state, done = env.step(action)
        
        data_controller.action_reciever(action)
        data_controller.move_idx()
        data_controller.state_reciever(np.array(list(state.values())))
        
        if done:
            x, y, reward = env.get_reward(state)
            data_controller.reward_reciever(reward)
            current_states, actions, next_states, rewards, terminals = data_controller.convert_data() # returns numpy array
            
            agent.update_network3(current_states, actions, next_states, rewards, terminals)
            
            plot_FIFO.add(x, y, reward)
            
            if agent.memory.n_entries >= agent.warmup:
                print("[Episode: %d]\n[Pruned Ratio: %f, Accuracy: %f, Reward: %f]\n[delta1: %f] [delta2: %f]" % ((j+1), x, y, reward, agent.delta1, agent.delta2))
                
                before_x, before_y, before_z = plot_FIFO.get_past()
                
                current_x, current_y, current_z = plot_FIFO.get_current()
                
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(before_x, before_y, before_z, s=1, marker="o")
                ax.scatter(current_x, current_y, current_z, s=40, marker="^")
                
                '''
                ax.set_xlim(opt.flop,1)
                ax.set_ylim(-env.max_loss,0)
                ax.set_zlim(-opt.flop*env.max_loss,0)
                '''
                
                plt.show()
                plt.close()
                print("="*100)
                
            finish = timeit.default_timer()
            print("Single Episode: %f" % (finish - start))
            
            state, done = env.reset()
            data_controller.state_reciever(np.array(list(state.values())))
            
            start = timeit.default_timer()

y_critic = agent.td_error_plot.tolist()
y_actor = agent.q_value_plot.tolist()
x_epi = list(range(len(y_critic)))

plt.plot(x_epi[5:], y_critic[5:], color='green', label = "TD-error")
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
plt.close()

plt.plot(x_epi[5:], y_actor[5:], color='red', label = "Q-value")
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
plt.close()


state, done = env.reset()
## ---- Validate pruned model ---- ##
for i in range(len(env.prunable_layer)):
    print("="*100)
    print(state)
    action = agent.deterministic_action2(np.array(list(state.values())), i, data_controller, True)
    state, done = env.step(action)
    
    if done:
        print("="*100)
        print(state)
        x, y, reward = env.get_reward(state)
        print("="*100)
        plot_FIFO.add(x, y, reward)
        before_x, before_y, before_z = plot_FIFO.get_past()
        current_x, current_y, current_z = plot_FIFO.get_current()
                
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(before_x, before_y, before_z, s=1, marker="o")
        ax.scatter(current_x, current_y, current_z, s=40, marker="^")
                                
        plt.show()
        plt.close()
        
        print("<Validation>")
        print("Pruned Model's Accuracy is %f" % y)
        #print("Pruned Model's FLOP Ratio is %f" % x)
        print("Pruned Model's Pruned Ratio is %f" % x)
        
        print("Pruned Model's Validation reward is %f" % reward)
        env.test(state)
        
env.original_result()

print(opt)













































