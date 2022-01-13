import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from scipy import stats
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, num_states, num_actions, hidden=1000):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_states, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, num_actions)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden=1000):
        super(Critic, self).__init__()
        self.fc1_1 = nn.Linear(num_states, hidden)
        self.fc1_2 = nn.Linear(num_actions, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, 1)
        self.relu = nn.LeakyReLU()
        
    def forward(self, state_action_list):
        state, action = state_action_list
        out = self.fc1_1(state) + self.fc1_2(action)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

class DDPG_Agent:
    def __init__(self, num_states, num_steps, options):
        ## ---- set basic information ---- ##
        self.num_states = num_states
        if options.method == 'both':
            self.num_actions = 3
        elif options.method == 'rectangular':
            self.num_actions = 2
        else:
            self.num_actions = 1
        
        self.num_steps = num_steps
        
        ## ---- build network ---- ##
        self.device = options.device
        
        self.actor = Actor(self.num_states, self.num_actions).to(self.device)
        self.actor_target = Actor(self.num_states, self.num_actions).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=options.lr)
        
        self.critic = Critic(self.num_states, self.num_actions).to(self.device)
        self.critic_target = Critic(self.num_states, self.num_actions).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=options.lr)
        
        self.critic_loss = nn.MSELoss()
        
        ## ---- instantiate memory ---- ##
        self.memory = Replay_Memory(self.num_states, self.num_actions, options.batch_size, options.memory_size)
        self.warmup = self.num_steps * options.warmup
        assert self.warmup <= options.memory_size, "Memory size should be bigger than warmup size"
        print("Warm up buffer size is", self.warmup)
        
        ## ---- initialize with same weight ---- ##
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        ## ---- Hyper parameter ---- ##
        self.init_delta = options.init_delta
        self.delta = self.init_delta
        self.delta_decay = options.delta_decay
        self.lbound = options.lbound
        self.rbound = options.rbound
        self.discount = options.discount
        self.tau = options.tau
        
        ## ---- Moving average ---- ##
        self.moving_average = None
        self.moving_alpha = 0.5
        
    def searching_action(self, episode, current_state):
        if len(self.memory.buffer) < self.warmup:
            action = self.refined_action(episode)
        else:
            tensor_state = torch.from_numpy(current_state)
            action = self.actor(tensor_state.float().to(self.device))
            action = action.clone().detach().cpu().numpy()
            
            self.delta = self.init_delta * (self.delta_decay ** (episode - math.floor(self.warmup/self.num_steps)))
            
            if self.num_actions == 3:
                action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=action[0], sigma=self.delta)
                action_2 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=action[1], sigma=self.delta)
                action_3 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=action[2], sigma=self.delta)
                
                action = np.array([action_1[0], action_2[0], action_3[0]])
                
            elif self.num_actions == 2:
                action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=action[0], sigma=self.delta)
                action_2 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=action[1], sigma=self.delta)
                
                action = np.array([action_1[0], action_2[0]])
                
            else:
                action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=action[0], sigma=self.delta)
                
                action = np.array([action_1[0]])
                
        return action
        
    def deterministic_action(self, current_state):
        self.actor.eval()
        tensor_state = torch.from_numpy(current_state)
        action = self.actor(tensor_state.float().to(self.device))
        action = action.clone().detach().cpu().numpy()
        print(action)
        return action
    
    def random_action(self):
        action = np.random.uniform(0,1,self.num_actions)
        return action
    
    def refined_action(self, episode):
        bound = ((episode+1) % 10)/10
        if ((episode+1) % 20) <= 10:
            action = np.random.uniform(bound,1,self.num_actions)
        else:
            action = np.random.uniform(0,bound,self.num_actions)
        return action
    
    def refined_action2(self, episode):
        bound = ((episode+1) % 10)/10
        if np.random.uniform(0,1,1)[0] <= 0.5:
            action = np.random.uniform(bound,1,self.num_actions)
        else:
            action = np.random.uniform(0,bound,self.num_actions)
        return action
    
    def update_memory_buffer(self, current_states, actions, next_states, rewards, terminals):
        for i in range(self.num_steps):
            self.memory.give_state_to_buffer(current_states[i], actions[i], next_states[i], rewards[i], terminals[i]) # type should be numpy array
        
    def update_network(self):
        
        assert len(self.memory.buffer) > self.warmup, "Buffer is not enough"
        a, b, c, d,e = self.memory.get_minibatch()
        CurStateBatch = a.float().to(self.device)
        ActionBatch = b.float().to(self.device)
        NextStateBatch = c.float().to(self.device)
        RewardBatch = d.float().to(self.device)
        TerminalBatch = e.float().to(self.device)
                
        ## ---- Normalize Reward ---- ##
        batch_mean_reward = RewardBatch.mean()
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
        RewardBatch -= self.moving_average
        
        ## ---- Update Critic ---- ##
        self.critic_optim.zero_grad()
        action_next = self.actor_target(NextStateBatch)
        Q_target_next = self.critic_target([NextStateBatch, action_next])
                
        Q_target = RewardBatch + self.discount * Q_target_next * TerminalBatch
        Q_expected = self.critic([CurStateBatch, ActionBatch])
        value_loss = self.critic_loss(Q_expected, Q_target)
        #print("Critic loss %f" % value_loss.item())
        value_loss.backward()
        self.critic_optim.step()
        
        ## ---- Update Actor ---- ##
        self.actor_optim.zero_grad()
        action_expected = self.actor(CurStateBatch)
        policy_loss = -self.critic([CurStateBatch, action_expected]).mean()
        #print("Actor loss is %f" % policy_loss)
        policy_loss.backward()
        self.actor_optim.step()
        
        ## ---- update target network ---- ##
        self.update_target_network()
    
    def update_target_network(self):
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)
        
    def soft_update(self, target, source):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                
    def sample_from_truncated_normal_distribution(self, lower, upper, mu, sigma, size=1):
        return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)
        
class Replay_Memory:
    def __init__(self, num_states, num_actions, batch_size, max_size=100000):
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.buffer = np.zeros((0))
        self.batch_size = batch_size
        self.max_size = max_size
        
    def give_state_to_buffer(self, current_state, action, next_state, reward, terminal):
        
        single_data = np.array([{'current_state':current_state,'action':action, 'next_state':next_state, 'reward':reward, 'terminal':terminal}])
        
        self.buffer = np.append(self.buffer, single_data, axis=0)
        
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[(len(self.buffer)-self.max_size):]
        
        if len(self.buffer) < self.batch_size:
            return False
        else:
            return True
    
    def get_minibatch(self):
        if len(self.buffer) < self.batch_size:
            print("Buffer is not enough to sample dataset. [%d/%d]" % (len(self.buffer), self.batch_size))
            return None, None, None, None
        
        else:
            sampled = self.buffer[np.random.choice(len(self.buffer), self.batch_size, replace=False)]
        
            current_state_batch = np.zeros((self.batch_size, self.num_states))
            action_batch = np.zeros((self.batch_size, self.num_actions))
            next_state_batch = np.zeros((self.batch_size, self.num_states))
            reward_batch = np.zeros((self.batch_size, 1))
            terminal_batch = np.zeros((self.batch_size, 1))
        
            for i in range(self.batch_size):
                current_state_batch[i] = sampled[i]['current_state']
                action_batch[i] = sampled[i]['action']
                next_state_batch[i] = sampled[i]['next_state']
                reward_batch[i] = sampled[i]['reward']
                terminal_batch[i] = sampled[i]['terminal']
        
            current_state_batch = torch.from_numpy(current_state_batch)
            action_batch = torch.from_numpy(action_batch)
            next_state_batch = torch.from_numpy(next_state_batch)
            reward_batch = torch.from_numpy(reward_batch)
            terminal_batch = torch.from_numpy(terminal_batch)
        
            return current_state_batch, action_batch, next_state_batch, reward_batch, terminal_batch
















