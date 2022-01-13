import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from scipy import stats
import torch.nn.functional as F
import matplotlib.pyplot as plt

'''
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
'''
class Actor(nn.Module):
    def __init__(self, num_states, num_actions, hidden=1000, middle=1000):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_states, hidden)
        self.fc2 = nn.Linear(hidden, middle)
        #self.fc3 = nn.Linear(middle, middle)
        self.fc4 = nn.Linear(middle, hidden)
        self.fc5 = nn.Linear(hidden, num_actions)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        #out = self.fc3(out)
        #out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.sigmoid(out)
        return out

class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden=1000, middle=1000):
        super(Critic, self).__init__()
        self.fc1_1 = nn.Linear(num_states, hidden)
        self.fc1_2 = nn.Linear(num_actions, hidden)
        self.fc2 = nn.Linear(hidden, middle)
        #self.fc3 = nn.Linear(middle, middle)
        self.fc4 = nn.Linear(middle, hidden)
        self.fc5 = nn.Linear(hidden, 1)
        self.relu = nn.LeakyReLU()
        
    def forward(self, state_action_list):
        state, action = state_action_list
        out = self.fc1_1(state) + self.fc1_2(action)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        #out = self.fc3(out)
        #out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out


'''
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
'''

class DDPG_Agent:
    def __init__(self, num_states, num_steps, options):
        ## ---- set basic information ---- ##
        self.num_states = num_states
        if options.method == 'both':
            self.num_actions = 3
        elif options.method == 'rectangular':
            self.num_actions = 2
        else: # channel
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
        
        ## ---- instantiate memory ---- ##
        self.batch_size = options.batch_size
        self.memory = Prioritized_Experience_Replay(self.num_states, self.num_actions, options.batch_size, options, options.memory_size)
        self.warmup = self.num_steps * options.warmup
        assert self.warmup <= options.memory_size, "Memory size should be bigger than warmup size"
        print("Warm up buffer size is", self.warmup)
        
        ## ---- initialize with same weight ---- ##
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        ## ---- Hyper parameter ---- ##
        self.init_delta1 = options.init_delta1
        self.init_delta2 = options.init_delta2
        self.delta1 = self.init_delta1
        self.delta2 = self.init_delta2
        self.delta_decay = options.delta_decay
        self.lbound = options.lbound
        self.rbound = options.rbound
        self.discount = options.discount
        self.tau = options.tau
        
        ## ---- Moving average ---- ##
        self.moving_average = None
        self.moving_alpha = 0.5
        
        ## ---- Action test ---- ##
        self.lbound_0, self.rbound_0 = 0, 0.05
        self.lbound_1, self.rbound_1 = 0, 0.05
        self.lbound_2, self.rbound_2 = 0, 0.05
        
        ## ---- loss plot ---- ##
        self.td_error_plot = np.zeros((0))
        self.q_value_plot = np.zeros((0))
    
    def searching_action(self, episode, current_state, current_state_idx, data_controller):
        if self.memory.n_entries < self.warmup:            
            action = self.random_action()
        else:
            embedded_state = data_controller.embed(current_state, current_state_idx)
            tensor_state = torch.from_numpy(embedded_state)
            assert (tensor_state > 1).sum() == 0, "Embedding is wrong"
            action = self.actor(tensor_state.float().to(self.device))
            action = action.clone().detach().cpu().numpy()
            
            self.delta1 = self.init_delta1 * (self.delta_decay ** (episode - math.floor(self.warmup/self.num_steps)))
            self.delta2 = self.init_delta2 * (self.delta_decay ** (episode - math.floor(self.warmup/self.num_steps)))
            
            if self.num_actions == 3:
                action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[0]), sigma=self.delta2)
                action_2 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[1]), sigma=self.delta2)
                action_3 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[2]), sigma=self.delta2)
                
                action = np.array([action_1[0], action_2[0], action_3[0]])
                
            elif self.num_actions == 2:
                action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[0]), sigma=self.delta2)
                action_2 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[1]), sigma=self.delta2)
                
                action = np.array([action_1[0], action_2[0]])
                
            else:
                action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[0]), sigma=self.delta2)
                
                action = np.array([action_1[0]])
            
        return action
    
    def searching_action2(self, episode, current_state, current_state_idx, data_controller):
        if self.memory.n_entries < self.warmup:
            self.var = ((episode+1) % 100)
            if self.var == 0:
                self.var = 100
            self.var = self.var/200
            
            for _ in range(current_state_idx):                
                self.var += 0.0001
            #print(self.var)
            action = self.truncated_random_action(self.var)
        else:
            embedded_state = data_controller.embed(current_state, current_state_idx)
            tensor_state = torch.from_numpy(embedded_state)
            assert (tensor_state > 1).sum() == 0, "Embedding is wrong"
            action = self.actor(tensor_state.float().to(self.device))
            action = action.clone().detach().cpu().numpy()
            #print("Action before %f" % action, end=" ")
            
            self.delta1 = self.init_delta1 * (self.delta_decay ** (episode - math.floor(self.warmup/self.num_steps)))
            self.delta2 = self.init_delta2 * (self.delta_decay ** (episode - math.floor(self.warmup/self.num_steps)))
            
            if self.num_actions == 3:
                action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[0]), sigma=self.delta2)
                action_2 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[1]), sigma=self.delta2)
                action_3 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[2]), sigma=self.delta2)
                
                action = np.array([action_1[0], action_2[0], action_3[0]])
                
            elif self.num_actions == 2:
                action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[0]), sigma=self.delta2)
                action_2 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[1]), sigma=self.delta2)
                
                action = np.array([action_1[0], action_2[0]])
                
            else:
                action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[0]), sigma=self.delta2)
                
                action = np.array([action_1[0]])
            
            #print("Action after %f" % action)
        return action
        
    def searching_action3(self, episode, current_state, current_state_idx, data_controller):
        if self.memory.n_entries < self.warmup:
            if ((episode+1) % 2) == 0:
                action = self.random_action()
            else:
                self.var = ((episode+1) % 100) # Hyperparameter
                if self.var == 0:
                    self.var = 100
                self.var = self.var/200 # Hyperparameter
                
                for _ in range(current_state_idx):                
                    self.var += 0.0001 # Hyperparameter
                #print(self.var)
                action = self.truncated_random_action(self.var)
        else:
            embedded_state = data_controller.embed(current_state, current_state_idx)
            tensor_state = torch.from_numpy(embedded_state)
            assert (tensor_state > 1).sum() == 0, "Embedding is wrong"
            action = self.actor(tensor_state.float().to(self.device))
            action = action.clone().detach().cpu().numpy()
            #print("Action before %f" % action, end=" ")
            
            self.delta1 = self.init_delta1 * (self.delta_decay ** (episode - math.floor(self.warmup/self.num_steps)))
            self.delta2 = self.init_delta2 * (self.delta_decay ** (episode - math.floor(self.warmup/self.num_steps)))
            
            if self.num_actions == 3:
                action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[0]), sigma=self.delta2)
                action_2 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[1]), sigma=self.delta2)
                action_3 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[2]), sigma=self.delta2)
                
                action = np.array([action_1[0], action_2[0], action_3[0]])
                
            elif self.num_actions == 2:
                action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[0]), sigma=self.delta2)
                action_2 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[1]), sigma=self.delta2)
                
                action = np.array([action_1[0], action_2[0]])
                
            else:
                action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=(self.delta1+action[0]), sigma=self.delta2)
                
                action = np.array([action_1[0]])
            
            #print("Action after %f" % action)
        return action   
    
    def deterministic_action(self, current_state, do_print=True):
        self.actor.eval()
        tensor_state = torch.from_numpy(current_state)
        action = self.actor(tensor_state.float().to(self.device))
        action = action.clone().detach().cpu().numpy()
        if do_print:
            print(action)
        return action
    
    def deterministic_action2(self, current_state, current_state_idx, data_controller, do_print=True):
        self.actor.eval()
        
        embedded_state = data_controller.embed(current_state, current_state_idx)
        tensor_state = torch.from_numpy(embedded_state)
        assert (tensor_state > 1).sum() == 0, "Embedding is wrong"
        action = self.actor(tensor_state.float().to(self.device))
        action = action.clone().detach().cpu().numpy()
        
        if do_print:
            print(action)
        
        return action
    
    def truncated_random_action(self, var):
        if self.num_actions == 3:
            action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=1, sigma=var)
            action_2 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=1, sigma=var)
            action_3 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=1, sigma=var)
                
            action = np.array([action_1[0], action_2[0], action_3[0]])
                
        elif self.num_actions == 2:
            action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=1, sigma=var)
            action_2 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=1, sigma=var)
                
            action = np.array([action_1[0], action_2[0]])
                
        else:
            action_1 = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=1, sigma=var)
                
            action = np.array([action_1[0]])
    
        return action
    
    def random_action(self):
        action = np.random.uniform(0,1,self.num_actions)
        return action
        
    def update_memory(self, current_state, action, next_state, reward, terminal, critic_priority, actor_priority):
        self.memory.add(current_state, action, next_state, reward, terminal, critic_priority, actor_priority)
        
    def update_network(self, current_states, actions, next_states, rewards, terminals): # input type is numpy array
        ## ---- Update Buffer ---- ##
        tensor_current_states = torch.from_numpy(current_states).float().to(self.device)
        tensor_actions = torch.from_numpy(actions).float().to(self.device)
        tensor_next_states = torch.from_numpy(next_states).float().to(self.device)
        tensor_rewards = torch.from_numpy(rewards).float().to(self.device)
        tensor_terminals = torch.from_numpy(terminals).float().to(self.device)
        
        for i in range(self.num_steps):
            ## ---- Critic priority ---- ##
            action_next = self.actor_target(tensor_next_states[i])
            Q_target_next = self.critic_target([tensor_next_states[i], action_next])
            Q_target = (tensor_rewards[i] + self.discount * Q_target_next * tensor_terminals[i])
            Q_expected = self.critic([tensor_current_states[i], tensor_actions[i]])
            
            td_error = Q_target - Q_expected
            
            ## ---- Actor priority ---- ##
            actor_action_expected = self.actor(tensor_current_states[i])
            
            Q_value = self.critic([tensor_current_states[i], actor_action_expected])
            
            ## ---- Update memory ---- ##
            self.update_memory(current_states[i], actions[i], next_states[i], rewards[i], terminals[i], td_error.abs(), Q_value.abs()) # store numpy in cpu
        
        ## ---- Update Network ---- ##
        if self.memory.n_entries < self.warmup:
            print("Buffer is currently being warmed up. [%d/%d]" % (self.memory.n_entries, self.warmup))
        
        else:
            ################ ---------------- Actor Critic with Prioritized Experience Replay ---------------- ################
            ######## -------- Critic -------- ########
            ## ---- get sample ---- ##
            critic_batch, critic_idx, critic_weight = self.memory.priority_sample(self.batch_size, True)

            critic_batch = np.array(critic_batch).transpose()
            
            critic_current_state = np.vstack(critic_batch[0])
            critic_action = np.vstack(critic_batch[1])
            critic_next_state = np.vstack(critic_batch[2])
            critic_reward = np.vstack(critic_batch[3])
            critic_terminal = np.vstack(critic_batch[4])
            
            print("="*100)
            plt.hist(critic_reward)
            plt.show()
            plt.close()
            
            critic_current_state = torch.from_numpy(critic_current_state).float().to(self.device)
            critic_action = torch.from_numpy(critic_action).float().to(self.device)
            critic_next_state = torch.from_numpy(critic_next_state).float().to(self.device)
            critic_reward = torch.from_numpy(critic_reward).float().to(self.device)
            critic_terminal = torch.from_numpy(critic_terminal).float().to(self.device)            
            
            ## ---- moving average ---- ##
            batch_mean_reward = critic_reward.mean().item()
            if self.moving_average is None:
                self.moving_average = batch_mean_reward
            else:
                self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
            critic_reward = critic_reward - self.moving_average
            
            ## ---- get priority: td error ---- ##
            with torch.no_grad():
                critic_action_next = self.actor_target(critic_next_state)
                critic_Q_target_next = self.critic_target([critic_next_state, critic_action_next])
            
            critic_Q_target = critic_reward + self.discount * critic_Q_target_next * critic_terminal
            critic_Q_expected = self.critic([critic_current_state, critic_action])

            td_error = critic_Q_target - critic_Q_expected
                
            ## ---- update priority ---- ##
            for i in range(self.batch_size):
                self.memory.update_priority(critic_idx[i], td_error.abs()[i], True)
                
            ## ---- update network ---- ##
            critic_weight = torch.from_numpy(critic_weight).float().to(self.device)
            
            self.critic_optim.zero_grad()
            critic_loss = (critic_weight * F.mse_loss(critic_Q_expected, critic_Q_target)).mean()
            self.critic_loss_plot = np.append(self.critic_loss_plot, np.array([critic_loss.item()]), axis=0)
            print("critic loss: %f" % critic_loss.item())
            critic_loss.backward()
            self.critic_optim.step()
            
            ######## -------- Actor  -------- ########
            self.actor_optim.zero_grad()
            
            ## ---- get priority: Q-value ---- ##
            actor_action_expected = self.actor(critic_current_state)
            Q_value = self.critic([critic_current_state, actor_action_expected])
            
            actor_loss = -Q_value.mean()
            self.actor_loss_plot = np.append(self.actor_loss_plot, np.array([actor_loss.item()]), axis=0)
            print("actor loss: %f" % actor_loss.item())
            actor_loss.backward()
            self.actor_optim.step()           
            
            ################ ---------------- update target network ---------------- ################
            self.update_target_network()
    
    def update_network2(self, current_states, actions, next_states, rewards, terminals): # input type is numpy array
        ## ---- Update Buffer ---- ##
        tensor_current_states = torch.from_numpy(current_states).float().to(self.device)
        tensor_actions = torch.from_numpy(actions).float().to(self.device)
        tensor_next_states = torch.from_numpy(next_states).float().to(self.device)
        tensor_rewards = torch.from_numpy(rewards).float().to(self.device)
        tensor_terminals = torch.from_numpy(terminals).float().to(self.device)
        
        ## ---- moving average ---- ##
        batch_mean_reward = tensor_rewards.mean().item()
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
        tensor_rewards = tensor_rewards - self.moving_average 
        
        for i in range(self.num_steps):
            ## ---- Critic priority ---- ##
            action_next = self.actor_target(tensor_next_states[i])
            Q_target_next = self.critic_target([tensor_next_states[i], action_next])
            Q_target = (tensor_rewards[i] + self.discount * Q_target_next * tensor_terminals[i])
            Q_expected = self.critic([tensor_current_states[i], tensor_actions[i]])
            
            td_error = Q_target - Q_expected
            
            ## ---- Actor priority ---- ##
            #actor_action_expected = self.actor(tensor_current_states[i])
            #Q_value = self.critic([tensor_current_states[i], actor_action_expected])
            
            ## ---- Update memory ---- ##
            self.update_memory(current_states[i], actions[i], next_states[i], rewards[i], terminals[i], td_error.abs(), rewards[i]) # store numpy in cpu
        
        ## ---- Update Network ---- ##
        if self.memory.n_entries < self.warmup:
            print("Buffer is currently being warmed up. [%d/%d]" % (self.memory.n_entries, self.warmup))
        
        else:
            ################ ---------------- Actor Critic with Prioritized Experience Replay ---------------- ################
            ######## -------- Critic -------- ########
            ## ---- get sample ---- ##
            critic_batch, critic_idx, critic_weight = self.memory.priority_sample(self.batch_size, True)

            critic_batch = np.array(critic_batch).transpose()
            
            critic_current_state = np.vstack(critic_batch[0])
            critic_action = np.vstack(critic_batch[1])
            critic_next_state = np.vstack(critic_batch[2])
            critic_reward = np.vstack(critic_batch[3])
            critic_terminal = np.vstack(critic_batch[4])
            
            print("="*100)
            plt.hist(critic_reward)
            plt.show()
            plt.close()
            
            critic_current_state = torch.from_numpy(critic_current_state).float().to(self.device)
            critic_action = torch.from_numpy(critic_action).float().to(self.device)
            critic_next_state = torch.from_numpy(critic_next_state).float().to(self.device)
            critic_reward = torch.from_numpy(critic_reward).float().to(self.device)
            critic_terminal = torch.from_numpy(critic_terminal).float().to(self.device)            
            
            ## ---- moving average ---- ##
            batch_mean_reward = critic_reward.mean().item()
            if self.moving_average is None:
                self.moving_average = batch_mean_reward
            else:
                self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
            critic_reward = critic_reward - self.moving_average
            
            ## ---- get priority: td error ---- ##
            with torch.no_grad():
                critic_action_next = self.actor_target(critic_next_state)
                critic_Q_target_next = self.critic_target([critic_next_state, critic_action_next])
            
            critic_Q_target = critic_reward + self.discount * critic_Q_target_next * critic_terminal
            critic_Q_expected = self.critic([critic_current_state, critic_action])

            td_error = critic_Q_target - critic_Q_expected
            self.td_error_plot = np.append(self.td_error_plot, np.array([td_error.abs().mean().item()]), axis=0)
            print("TD-error: %f" % td_error.abs().mean().item())
            
            ## ---- update priority ---- ##
            for i in range(self.batch_size):
                self.memory.update_priority(critic_idx[i], td_error.abs()[i], True)
                
            ## ---- update network ---- ##
            critic_weight = torch.from_numpy(critic_weight).float().to(self.device)
            
            self.critic_optim.zero_grad()
            critic_loss = (critic_weight * F.mse_loss(critic_Q_expected, critic_Q_target)).mean()
            critic_loss.backward()
            self.critic_optim.step()
            
            ######## -------- Actor  -------- ########
            ## ---- get sample ---- ##
            actor_batch, actor_idx, actor_weight = self.memory.priority_sample(self.batch_size, False)
            
            actor_batch = np.array(actor_batch).transpose()
            
            actor_current_state = np.vstack(actor_batch[0])
            actor_reward = np.vstack(actor_batch[3])
            
            print("="*100)
            plt.hist(actor_reward)
            plt.show()
            plt.close()
            
            actor_current_state = torch.from_numpy(actor_current_state).float().to(self.device)

            ## ---- get priority: Q-value ---- ##
            actor_action_expected = self.actor(actor_current_state)
            Q_value = self.critic([actor_current_state, actor_action_expected])
            
            self.q_value_plot = np.append(self.q_value_plot, np.array([Q_value.mean().item()]), axis=0)
            print("Q-value: %f" % Q_value.mean().item())
            
            ## ---- update priority ---- ##
            #for i in range(self.batch_size):    
            #    self.memory.update_priority(actor_idx[i], 1/td_error.abs()[i], False)            
            
            ## ---- update network ---- ##
            actor_weight = torch.from_numpy(actor_weight).float().to(self.device)
            
            self.actor_optim.zero_grad()
            actor_loss = -(actor_weight * Q_value).mean()
            actor_loss.backward()
            self.actor_optim.step()           
            
            ################ ---------------- update target network ---------------- ################
            self.update_target_network()
    
    def update_network3(self, current_states, actions, next_states, rewards, terminals): # input type is numpy array
        ## ---- Update Buffer ---- ##
        tensor_current_states = torch.from_numpy(current_states).float().to(self.device)
        tensor_actions = torch.from_numpy(actions).float().to(self.device)
        tensor_next_states = torch.from_numpy(next_states).float().to(self.device)
        tensor_rewards = torch.from_numpy(rewards).float().to(self.device)
        tensor_terminals = torch.from_numpy(terminals).float().to(self.device)
        
        ## ---- moving average ---- ##
        batch_mean_reward = tensor_rewards.mean().item()
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
        tensor_rewards = tensor_rewards - self.moving_average        
        
        for i in range(self.num_steps):
            ## ---- Critic priority ---- ##
            action_next = self.actor_target(tensor_next_states[i])
            Q_target_next = self.critic_target([tensor_next_states[i], action_next])
            Q_target = (tensor_rewards[i] + self.discount * Q_target_next * tensor_terminals[i])
            Q_expected = self.critic([tensor_current_states[i], tensor_actions[i]])
            
            td_error = Q_target - Q_expected
            
            ## ---- Actor priority ---- ##
            actor_action_expected = self.actor(tensor_current_states[i])
            
            Q_value = self.critic([tensor_current_states[i], actor_action_expected])
            
            ## ---- Update memory ---- ##
            self.update_memory(current_states[i], actions[i], next_states[i], rewards[i], terminals[i], td_error.abs(), Q_value.abs()) # store numpy in cpu
        
        ## ---- Update Network ---- ##
        if self.memory.n_entries < self.warmup:
            print("Buffer is currently being warmed up. [%d/%d]" % (self.memory.n_entries, self.warmup))
        
        else:
            ################ ---------------- Actor Critic with Prioritized Experience Replay ---------------- ################
            ######## -------- Critic -------- ########
            ## ---- get sample ---- ##
            critic_batch, critic_idx, critic_weight = self.memory.priority_sample(self.batch_size, True)

            critic_batch = np.array(critic_batch).transpose()
            
            critic_current_state = np.vstack(critic_batch[0])
            critic_action = np.vstack(critic_batch[1])
            critic_next_state = np.vstack(critic_batch[2])
            critic_reward = np.vstack(critic_batch[3])
            critic_terminal = np.vstack(critic_batch[4])
            
            print("="*100)
            plt.hist(critic_reward)
            plt.show()
            plt.close()
            
            critic_current_state = torch.from_numpy(critic_current_state).float().to(self.device)
            critic_action = torch.from_numpy(critic_action).float().to(self.device)
            critic_next_state = torch.from_numpy(critic_next_state).float().to(self.device)
            critic_reward = torch.from_numpy(critic_reward).float().to(self.device)
            critic_terminal = torch.from_numpy(critic_terminal).float().to(self.device)            
            
            ## ---- moving average ---- ##
            batch_mean_reward = critic_reward.mean().item()
            if self.moving_average is None:
                self.moving_average = batch_mean_reward
            else:
                self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
            critic_reward = critic_reward - self.moving_average
            
            ## ---- get priority: td error ---- ##
            with torch.no_grad():
                critic_action_next = self.actor_target(critic_next_state)
                critic_Q_target_next = self.critic_target([critic_next_state, critic_action_next])
            
            critic_Q_target = critic_reward + self.discount * critic_Q_target_next * critic_terminal
            critic_Q_expected = self.critic([critic_current_state, critic_action])

            td_error = critic_Q_target - critic_Q_expected
            self.td_error_plot = np.append(self.td_error_plot, np.array([td_error.abs().mean().item()]), axis=0)
            print("TD-error: %f" % td_error.abs().mean().item())
            
            ## ---- update priority ---- ##
            for i in range(self.batch_size):
                self.memory.update_priority(critic_idx[i], td_error.abs()[i], True)
                
            ## ---- update network ---- ##
            critic_weight = torch.from_numpy(critic_weight).float().to(self.device)
            
            self.critic_optim.zero_grad()
            critic_loss = (critic_weight * F.mse_loss(critic_Q_expected, critic_Q_target)).mean()
            critic_loss.backward()
            self.critic_optim.step()
            
            ######## -------- Actor  -------- ########
            ## ---- get priority: Q-value ---- ##
            actor_action_expected = self.actor(critic_current_state)
            Q_value = self.critic([critic_current_state, actor_action_expected])
            
            self.q_value_plot = np.append(self.q_value_plot, np.array([Q_value.mean().item()]), axis=0)
            print("Q-value: %f" % Q_value.mean().item())
            
            self.actor_optim.zero_grad()
            actor_loss = -Q_value.mean()
            actor_loss.backward()
            self.actor_optim.step()           
            
            ################ ---------------- update target network ---------------- ################
            self.update_target_network()
            
    def update_network4(self, current_states, actions, next_states, rewards, terminals): # input type is numpy array
        ## ---- Update Buffer ---- ##
        tensor_current_states = torch.from_numpy(current_states).float().to(self.device)
        tensor_actions = torch.from_numpy(actions).float().to(self.device)
        tensor_next_states = torch.from_numpy(next_states).float().to(self.device)
        tensor_rewards = torch.from_numpy(rewards).float().to(self.device)
        tensor_terminals = torch.from_numpy(terminals).float().to(self.device)
        
        ## ---- moving average ---- ##
        batch_mean_reward = tensor_rewards.mean().item()
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
        tensor_rewards = tensor_rewards - self.moving_average        
        
        ## ---- Critic priority ---- ##
        action_next = self.actor_target(tensor_next_states)
        Q_target_next = self.critic_target([tensor_next_states, action_next])
        Q_target = (tensor_rewards + self.discount * Q_target_next * tensor_terminals)
        Q_expected = self.critic([tensor_current_states, tensor_actions])
        
        td_error = Q_target - Q_expected
            
        ## ---- Actor priority ---- ##
        actor_action_expected = self.actor(tensor_current_states)
            
        Q_value = self.critic([tensor_current_states, actor_action_expected])
        
        for i in range(self.num_steps):
            ## ---- Update memory ---- ##
            self.update_memory(current_states[i], actions[i], next_states[i], rewards[i], terminals[i], td_error.abs()[i], Q_value.abs()[i]) # store numpy in cpu
        
        ## ---- Update Network ---- ##
        if self.memory.n_entries < self.warmup:
            print("Buffer is currently being warmed up. [%d/%d]" % (self.memory.n_entries, self.warmup))
        
        else:
            ################ ---------------- Actor Critic with Prioritized Experience Replay ---------------- ################
            ######## -------- Critic -------- ########
            ## ---- get sample ---- ##
            critic_batch, critic_idx, critic_weight = self.memory.priority_sample(self.batch_size, True)

            critic_batch = np.array(critic_batch).transpose()
            
            critic_current_state = np.vstack(critic_batch[0])
            critic_action = np.vstack(critic_batch[1])
            critic_next_state = np.vstack(critic_batch[2])
            critic_reward = np.vstack(critic_batch[3])
            critic_terminal = np.vstack(critic_batch[4])
            
            print("="*100)
            plt.hist(critic_reward)
            plt.show()
            plt.close()
            
            critic_current_state = torch.from_numpy(critic_current_state).float().to(self.device)
            critic_action = torch.from_numpy(critic_action).float().to(self.device)
            critic_next_state = torch.from_numpy(critic_next_state).float().to(self.device)
            critic_reward = torch.from_numpy(critic_reward).float().to(self.device)
            critic_terminal = torch.from_numpy(critic_terminal).float().to(self.device)            
            
            ## ---- moving average ---- ##
            batch_mean_reward = critic_reward.mean().item()
            if self.moving_average is None:
                self.moving_average = batch_mean_reward
            else:
                self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
            critic_reward = critic_reward - self.moving_average
            
            ## ---- get priority: td error ---- ##
            with torch.no_grad():
                critic_action_next = self.actor_target(critic_next_state)
                critic_Q_target_next = self.critic_target([critic_next_state, critic_action_next])
            
            critic_Q_target = critic_reward + self.discount * critic_Q_target_next * critic_terminal
            critic_Q_expected = self.critic([critic_current_state, critic_action])

            td_error = critic_Q_target - critic_Q_expected
            self.td_error_plot = np.append(self.td_error_plot, np.array([td_error.abs().mean().item()]), axis=0)
            print("TD-error: %f" % td_error.abs().mean().item())
            
            ## ---- update priority ---- ##
            for i in range(self.batch_size):
                self.memory.update_priority(critic_idx[i], td_error.abs()[i], True)
                
            ## ---- update network ---- ##
            critic_weight = torch.from_numpy(critic_weight).float().to(self.device)
            
            self.critic_optim.zero_grad()
            critic_loss = (critic_weight * F.mse_loss(critic_Q_expected, critic_Q_target)).mean()
            critic_loss.backward()
            self.critic_optim.step()
            
            ######## -------- Actor  -------- ########
            ## ---- get priority: Q-value ---- ##
            actor_action_expected = self.actor(critic_current_state)
            Q_value = self.critic([critic_current_state, actor_action_expected])
            
            self.q_value_plot = np.append(self.q_value_plot, np.array([Q_value.mean().item()]), axis=0)
            print("Q-value: %f" % Q_value.mean().item())
            
            self.actor_optim.zero_grad()
            actor_loss = -Q_value.mean()
            actor_loss.backward()
            self.actor_optim.step()           
            
            ################ ---------------- update target network ---------------- ################
            self.update_target_network()
    
    def update_target_network(self):
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)
        
    def soft_update(self, target, source):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                
    def sample_from_truncated_normal_distribution(self, lower, upper, mu, sigma, size=1):
        return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)

class Prioritized_Experience_Replay:
    def __init__(self, num_states, num_actions, batch_size, options, capacity=1000000):
        self.num_states = num_states
        self.num_actions = num_actions
        self.capacity = capacity
        
        self.buffer = np.zeros(self.capacity, dtype=object)
        self.critic_priorities = np.zeros(self.capacity)
        self.actor_priorities = np.zeros(self.capacity)
        
        if self.isPowerOfTwo(self.capacity):
            self.critic_tree = np.zeros(2 * self.capacity - 1)
            self.actor_tree = np.zeros(2 * self.capacity - 1)
            self.tree_type = True
            
        else:
            self.critic_tree = np.zeros(2 ** (math.ceil(math.log2(self.capacity)) + 1) - 1)
            self.actor_tree = np.zeros(2 ** (math.ceil(math.log2(self.capacity)) + 1) - 1)
            self.tree_type = False
            
        self.buffer_idx = 0
        self.n_entries = 0 # How many transitions are stored in buffer
        
        self.beta = options.beta
        self.beta_increment_per_sampling = options.beta_increment_per_sampling
        self.epsilon = options.epsilon
        self.alpha = options.alpha
        
    def isPowerOfTwo(self, x):
        return (x and (not(x & (x - 1))))
    
    def add(self, current_state, action, next_state, reward, terminal, critic_priority, actor_priority):
        transition = [current_state, action, next_state, reward, terminal]
        self.buffer[self.buffer_idx] = transition
        self.critic_priorities[self.buffer_idx] = self._get_priority(critic_priority)
        self.actor_priorities[self.buffer_idx] = self._get_priority(actor_priority)
        
        if self.tree_type:
            self.critic_tree_idx = self.buffer_idx + self.capacity - 1
            self.actor_tree_idx = self.buffer_idx + self.capacity - 1
            
        else:
            self.critic_tree_idx = self.buffer_idx + (2 ** math.ceil(math.log2(self.capacity))) - 1
            self.actor_tree_idx = self.buffer_idx + (2 ** math.ceil(math.log2(self.capacity))) - 1
        
        self.update_priority(self.critic_tree_idx, critic_priority, True)
        self.update_priority(self.actor_tree_idx, actor_priority, False)
        
        self.buffer_idx += 1
        
        if self.buffer_idx >= self.capacity: # First in, First out
            self.buffer_idx = 0            
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
        
    def _get_priority(self, value):
        return (abs(value.item()) + self.epsilon) ** self.alpha # proportional variant
    
    def update_priority(self, index, value, critic=True):
        new_priority = self._get_priority(value)
        
        if critic:
            change = new_priority - self.critic_tree[index]
            self.critic_tree[index] = new_priority
            self._propagate(index, change, critic)
        else:
            change = new_priority - self.actor_tree[index]
            self.actor_tree[index] = new_priority
            self._propagate(index , change, critic)
        
    def _propagate(self, index, change, critic=True):
        parent = (index - 1) // 2
        
        if critic:
            self.critic_tree[parent] += change
        else:
            self.actor_tree[parent] += change

        if parent != 0:
            self._propagate(parent, change, critic)
        
    def priority_sample(self, number, critic=True):
        
        batch = []
        idxs = []
        priorities = []
        if critic:
            segment = self.critic_tree[0] / number
        else:
            segment = self.actor_tree[0] / number
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(number):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            idx, p, data = self.get(s, critic)
            idxs.append(idx)
            priorities.append(p)
            batch.append(data)
        
        if critic:
            sampling_probabilities = np.array([x / self.critic_tree[0] for x in priorities])
        else:
            sampling_probabilities = np.array([x / self.actor_tree[0] for x in priorities])
        
        is_weight = np.power(self.n_entries * sampling_probabilities, -self.beta)
        
        is_weight /= is_weight.max()
        
        return batch, idxs, is_weight
        
    def get(self, s, critic=True):
        
        idx = self._retrieve(0, s, critic)
        
        if self.tree_type:
            dataIdx = idx - self.capacity + 1
        else:
            dataIdx = idx - (2 ** math.ceil(math.log2(self.capacity))) + 1
        
        if critic:
            return idx, self.critic_tree[idx], self.buffer[dataIdx]
        else:
            return idx, self.actor_tree[idx], self.buffer[dataIdx]
        
    def _retrieve(self, idx, s, critic=True):
        
        left = 2 * idx + 1
        right = left + 1
        
        if critic:
            if left >= len(self.critic_tree):
                return idx

            if s <= self.critic_tree[left]:
                return self._retrieve(left, s, critic)
            else:
                return self._retrieve(right, s - self.critic_tree[left], critic)
            
        else:
            if left >= len(self.actor_tree):
                return idx

            if s <= self.actor_tree[left]:
                return self._retrieve(left, s, critic)
            else:
                return self._retrieve(right, s - self.actor_tree[left], critic)


























