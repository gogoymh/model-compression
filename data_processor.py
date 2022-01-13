import numpy as np


class processor:
    def __init__(self, num_states, num_steps, options):
        self.num_states = num_states
        self.num_steps = num_steps
        
        if options.method == 'both':
            self.num_actions = 3
        elif options.method == 'rectangular':
            self.num_actions = 2
        else:
            self.num_actions = 1
        
        self.states = np.zeros(((num_steps + 1), num_states))
        self.actions = np.zeros((num_steps, self.num_actions))
        self.rewards = np.zeros((num_steps, 1))
        self.terminals = np.ones((num_steps, 1))
        self.terminals[-1] = 0
        
        self.current_states_idx = 0
        
    def reset_idx(self):
        self.current_states_idx = 0
    
    def state_reciever(self, state): # state는 dictionary지만 numpy array로 변경시켜서 집어넣을 것이다.
        self.states[self.current_states_idx] = state
        
    def action_reciever(self, action):
        self.actions[self.current_states_idx] = action
        
    def reward_reciever(self, reward):
        self.rewards[:] = reward
    
    def move_idx(self):
        self.current_states_idx += 1
    
    def convert_data(self):
        for i in range(self.num_states):
            smin = self.states[:,i].min()
            smax = self.states[:,i].max()
            if smin == smax:
                self.states[:,i] = max(min(1, smin), 0)
                #print('%d is useless information' % i)
            else:
                self.states[:,i] = (self.states[:,i] - smin)/(smax - smin)
        
        self.current_states = self.states[:(self.num_steps + 1)]
        self.next_states = self.states[1:]
        
        self.current_states_idx = 0 # reset
        
        return self.current_states.copy(), self.actions.copy(), self.next_states.copy(), self.rewards.copy(), self.terminals.copy()