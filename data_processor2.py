import numpy as np


class processor:
    def __init__(self, num_states, num_steps, total_FLOPs, options):
        self.num_states = num_states
        self.num_steps = num_steps
        self.total_FLOPs = total_FLOPs
        
        if options.method == 'both':
            self.num_actions = 3
        elif options.method == 'rectangular':
            self.num_actions = 2
        else:
            self.num_actions = 1
        
        ## ---- empty array ---- ##
        self.states = np.zeros(((num_steps + 1), num_states))
        self.actions = np.zeros((num_steps, self.num_actions))
        self.rewards = np.zeros((num_steps, 1))
        self.terminals = np.ones((num_steps, 1))
        self.terminals[-1] = 0
        
        ## ---- current state ---- ##
        self.current_states_idx = 0
        
        ## ---- embedding information ---- ##
        self.smin = np.zeros((15, 1))
        self.smax = np.zeros((15, 1))
        self.dynamic_info_value = np.zeros(((num_steps + 1), 2))
        
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
        for j in range(self.num_steps+1):
            self.states[j,15] = self.states[j,15]/self.dynamic_info_value[j,0] if self.dynamic_info_value[j,0] != 0 else self.states[j,15]
            self.states[j,16] = self.states[j,16]/self.dynamic_info_value[j,1] if self.dynamic_info_value[j,1] != 0 else self.states[j,16]
            self.states[j,17] = self.states[j,17]/self.dynamic_info_value[j,0] if self.dynamic_info_value[j,0] != 0 else self.states[j,17]
            self.states[j,18] = self.states[j,18]/self.dynamic_info_value[j,1] if self.dynamic_info_value[j,1] != 0 else self.states[j,18]
            
        self.states[:,19] /= self.total_FLOPs
        
        for i in range(0, 15):
            smin = self.smin[i]
            smax = self.smax[i]
            
            if smin == smax:
                self.states[:,i] = max(min(1, smin), 0)
                #print('%d is useless information' % i)
            else:
                self.states[:,i] = (self.states[:,i] - smin)/(smax - smin)
        
        self.current_states = self.states[:self.num_steps]
        self.next_states = self.states[1:]
        
        self.reset_idx()
        
        return self.current_states.copy(), self.actions.copy(), self.next_states.copy(), self.rewards.copy(), self.terminals.copy()
    
    def embed(self, state, state_idx): # state's type is numpy array
        embedded_state = state.copy()
        for i in range(0, 15):
            smin = self.smin[i]
            smax = self.smax[i]
            
            if smin == smax:
                embedded_state[i] = max(min(1, smin), 0)
            else:
                embedded_state[i] = (embedded_state[i] - smin)/(smax - smin)
        
        embedded_state[15] = embedded_state[15]/self.dynamic_info_value[state_idx,0] if self.dynamic_info_value[state_idx,0] != 0 else embedded_state[15]
        embedded_state[16] = embedded_state[16]/self.dynamic_info_value[state_idx,1] if self.dynamic_info_value[state_idx,1] != 0 else embedded_state[16]
        embedded_state[17] = embedded_state[17]/self.dynamic_info_value[state_idx,0] if self.dynamic_info_value[state_idx,0] != 0 else embedded_state[17]
        embedded_state[18] = embedded_state[18]/self.dynamic_info_value[state_idx,1] if self.dynamic_info_value[state_idx,1] != 0 else embedded_state[18]
        embedded_state[19] /= self.total_FLOPs
        
        return embedded_state
    
    def get_embed_info(self, environment):
        state, done = environment.reset()
        self.state_reciever(np.array(list(state.values())))
        
        while done == 0:
            action = np.ones((self.num_actions))
            state, done = environment.step(action)
            
            self.action_reciever(action)
            self.move_idx()
            self.state_reciever(np.array(list(state.values())))
            
        ## ---- static information ---- ##
        for i in range(0, 15):
            self.smin[i] = self.states[:,i].min()
            self.smax[i] = self.states[:,i].max()
        
        #3 ---- dynamic information ---- ##
        for j in range(self.num_steps+1):
            self.dynamic_info_value[j,0] = self.states[j,12]
            self.dynamic_info_value[j,1] = self.states[j,13]
        
        ## ---- reset ---- ##
        self.reset_idx()
        state, done = environment.reset()

        return state, done














