from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import argparse
import numpy as np
#import timeit

from network import resnet50 as ResNet
#from network2 import ResNet18 as ResNet
from Environment3 import Environment
from Agent import DDPG_Agent
from data_processor import processor

###########################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
parser.add_argument("--method", type=str, default="both", help="Method")
parser.add_argument("--init_delta", type=float, default=1, help="Initial delta")
parser.add_argument("--delta_decay", type=float, default=0.95, help="Delta decay")
parser.add_argument("--lbound", type=float, default=0, help="Left action bound")
parser.add_argument("--rbound", type=float, default=1, help="Right action bound")
parser.add_argument("--discount", type=float, default=1, help="Discount factor")
parser.add_argument("--tau", type=float, default=0.01, help="Tau for soft update")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--memory_size", type=int, default=1500, help="Memory size")
parser.add_argument("--warmup", type=int, default=20, help="Warm up episodes")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--episodes", type=int, default=100000, help="Total episodes")

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
data_holder = processor(len(state), len(env.prunable_layer), opt)

data_holder.state_reciever(np.array(list(state.values())))


pair = np.zeros((100000))
import matplotlib.pyplot as plt
## ---- Search ---- ##
for j in range(opt.episodes):
    #start = timeit.default_timer()
    for i in range(len(env.prunable_layer)):
        action = agent.refined_action(j)
        state, done = env.step(action)
        
        data_holder.action_reciever(action)
        data_holder.move_idx()
        data_holder.state_reciever(np.array(list(state.values())))
        
        if done:
            pair[j] = env.get_reward(state, 0.8, 0.7, 0.2, 1e-6)
            
            data_holder.reset_idx()
    
            '''
            data_holder.reward_reciever(reward)
            a, b, c, d, e = data_holder.convert_data()
            agent.update_memory_buffer(a,b,c,d,e)
            
            if len(agent.memory.buffer) > agent.warmup:
                agent.update_network()            
                _, _, _, reward_batch, _ = agent.memory.get_minibatch()
                    
                print("="*100)
                print("[Episode: %d] [Reward: %f] [delta: %f] [Mean Reward Batch: %f]" % ((j+1), reward, agent.delta, reward_batch.mean().item()))
            else:
                print("Updating main network is skipped. Buffer is warming up. [%d/%d]" % (len(agent.memory.buffer), agent.warmup))
            
            #finish = timeit.default_timer()
            #print("Single Episode: %f" % (finish - start))
            '''
            if (j+1) % 10 == 0:
                print("=" * 100)
                print(j+1)
                
            if (j+1) % 50 == 0:
                plt.hist(pair[:(j+1)], bins=30)
                plt.show()
                plt.close()
            
            
            state, done = env.reset()
            
            data_holder.state_reciever(np.array(list(state.values())))
            #start = timeit.default_timer()




'''
## ---- Validate pruned model ---- ##
for i in range(len(env.prunable_layer)):
    print("="*100)
    print(state)
    action = agent.deterministic_action(np.array(list(state.values())))
    state, done = env.step(action)
    
    data_holder.action_reciever(action)
    data_holder.move_idx()
    data_holder.state_reciever(np.array(list(state.values())))
    
    if done:
        print("="*100)
        print(state)
        env.test(state)
        
env.original_result()
'''














































