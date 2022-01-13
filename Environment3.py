import torch
import copy
import numpy as np
import torch.nn as nn
import math

class Environment:
    def __init__(self, model, path, test_loader, options):
        
        ## ---- save method ---- ##
        self.method = options.method
        
        ## ---- prepare dataset ---- ##
        self.validation_set = []
        self.test_set = []
        self.half_length = int(len(test_loader.dataset)/2)
        for batch_idx, data in enumerate(test_loader):
            if batch_idx < 50:
                self.validation_set.append(data)
            else:
                self.test_set.append(data)
        print("Validation/Test set length is both", self.half_length)
        
        ## ---- build model ---- ##
        self.device = options.device
        self.model = model().to(self.device)
        
        if self.device == "cuda:0":
            self.checkpoint = torch.load(path)
        else:
            self.checkpoint = torch.load(path, map_location=lambda storage, location: 'cpu')
        
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        
        ## ---- prepare layers to deal with ---- ##
        self.module_list = list(self.model.modules())
        self.prunable_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]
        
        self.prunable_layer = []
        for i, m in enumerate(self.module_list):
            if type(m) in self.prunable_layer_types:
                if type(m) == nn.Conv2d and m.groups == m.in_channels:  # depth-wise conv, buffer
                    pass
                else:
                    self.prunable_layer.append(i)
                
        print('Prunable layer is', self.prunable_layer)
        
        ## ---- define states ---- ##
        def extract_information_from_layer(layer, x):
            def get_layer_type(layer):
                layer_str = str(layer)
                return layer_str[:layer_str.find('(')].strip()
            
            type_name = get_layer_type(layer)
            
            if type_name in ['Conv2d']:
                layer.input_height = x.size()[2]
                layer.input_width = x.size()[3]
                layer.output_height = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) / layer.stride[0] + 1)
                layer.output_width = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) / layer.stride[1] + 1)
                layer.flops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * layer.output_height * layer.output_width / layer.groups
                layer.type = 0 # Convolutional
                
            elif type_name in ['Linear']:
                layer.input_height = 1
                layer.input_width = 1
                layer.output_height = 1 
                layer.output_width = 1
                
                weight_ops = layer.weight.numel()
                bias_ops = layer.bias.numel()
                layer.flops = weight_ops + bias_ops
                
                layer.stride = 0, 0
                layer.kernel_size = 1, 1
                layer.in_channels = layer.in_features
                layer.out_channels = layer.out_features
                layer.type = 1 # Linear
                layer.padding = 0, 0
                
            return
        
        def new_forward(m): # 이름 바꾸기
            def lambda_forward(x): # 
                extract_information_from_layer(m, x)
                y = m.old_forward(x)
                return y
            
            return lambda_forward
        
        for idx in self.prunable_layer:  # get all
            m = self.module_list[idx]
            m.old_forward = m.forward
            m.forward = new_forward(m)
        
        with torch.no_grad():
            self.model.eval()
            correct = 0
            for x, y in self.validation_set:
                output = self.model(x.float().to(self.device))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(self.device).view_as(pred)).sum().item()
        
        self.accuracy_original = correct/self.half_length
        
        def embed_state_from_information(self, layer_idx, final=False):
            
            m = self.module_list[layer_idx]
                
            if final:
                ## ---- There isn't kernel ---- ##
            
                kernel_index = self.current_state_idx
                kernel_out_channels = m.out_channels
                kernel_height = 0
                kernel_width = 0
                stride_height = 0
                stride_width = 0
                padding_height = 0
                padding_width = 0
                FLOPs = 0
                kernel_type = m.type
                
                ## ---- output feature information ---- ##
                
                feature_height = m.output_height
                feature_width = m.output_width
                feature_channel = kernel_out_channels
                
            else:
                ## ---- kernel information ---- ##
            
                kernel_index = self.current_state_idx
                kernel_out_channels = m.out_channels
                kernel_height = m.kernel_size[0]
                kernel_width = m.kernel_size[1]
                stride_height = m.stride[0]
                stride_width = m.stride[1]
                padding_height = m.padding[0]
                padding_width = m.padding[1]
                FLOPs = m.flops
                kernel_type = m.type
                
                ## ---- input feature information ---- ##
            
                feature_height = m.input_height
                feature_width = m.input_width
                feature_channel = m.in_channels
                
            ## ---- pseudo response information ---- ##
            Reduced = 0
            previous_height = 0
            previous_width = 0
            x = 0
            y = 0
            receptive_height = 0
            receptive_width = 0
            
            ## ---- make dictionary for state ---- ##
            layer_info = dict()
            ## -- static information -- ##
            layer_info['feature_height'] = feature_height
            layer_info['feature_width'] =  feature_width
            layer_info['feature_channel'] = feature_channel
            layer_info['kernel_index'] = kernel_index
            layer_info['kernel_type'] = kernel_type
            layer_info['kernel_out_channels'] = kernel_out_channels
            layer_info['kernel_height'] = kernel_height
            layer_info['kernel_width'] = kernel_width
            layer_info['stride_height'] = stride_height
            layer_info['stride_width'] = stride_width
            layer_info['padding_height'] = padding_height
            layer_info['padding_width'] = padding_width
            layer_info['previous_height'] = previous_height
            layer_info['previous_width'] = previous_width
            layer_info['FLOPs'] = int(FLOPs)
            
            ## -- dynamic information(depends on action) -- ##
            layer_info['x'] = x
            layer_info['y'] = y
            layer_info['receptive_height'] = receptive_height
            layer_info['receptive_width'] = receptive_width
            layer_info['Reduced'] = Reduced
            
            return layer_info
        
        self.basic_states = []
        self.current_state_idx = 0
        for i in self.prunable_layer:
            self.basic_states.append(embed_state_from_information(self, layer_idx=i, final=False))
            self.current_state_idx += 1
        self.basic_states.append(embed_state_from_information(self, layer_idx=i, final=True))
        
        self.total_FLOPs = 0
        for j in range(len(self.basic_states)):
            self.total_FLOPs += self.basic_states[j]['FLOPs']
        print('Total FLOP is %d' % self.total_FLOPs)
        
    def reset(self):
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        
        self.episodic_states = copy.deepcopy(self.basic_states)
        self.current_state_idx = 0
        initial_state = self.episodic_states[self.current_state_idx]
        
        return initial_state, False
        
    def step(self, action):
        ## ---- get current state and kernel ---- ##
        current_state = self.episodic_states[self.current_state_idx]
        current_kernel = self.module_list[self.prunable_layer[self.current_state_idx]]
        weight = current_kernel.weight.data.clone().cpu().numpy()
        
        if self.method == 'channel':
            assert len(action) == 1, "Channel pruning action should be single float number"
            
            ## ---- execute action ---- ##
            num_prune_c = int(round((1 - action[0]) * current_state['feature_channel']))
            if current_state['kernel_type'] == 0: # Convolutional
                ## ---- prune channel ---- ##
                importance_c = np.abs(weight).sum((0,2,3))
                sorted_idx_c = np.argsort(importance_c)
                preserve_idx_c = sorted_idx_c[num_prune_c:] if num_prune_c != current_state['feature_channel'] else range(0, weight.shape[1])
                mask_c = np.ones(weight.shape[1], bool)
                mask_c[preserve_idx_c] = False
                weight[:,mask_c,:,:] = 0
                
                ## ---- update information ---- ##
                if num_prune_c == 0:
                    C_ratio = current_state['feature_channel']
                    FLOPs = 0
                else:
                    C_ratio = num_prune_c
                    FLOPs = current_state['FLOPs']
                Reduced = current_state['Reduced'] + (C_ratio/current_state['feature_channel']) * FLOPs
                previous_height = current_state['feature_height']
                previous_width = current_state['feature_width']
                x = 0
                y = 0
                receptive_height = current_state['feature_height']
                receptive_width = current_state['feature_width']
                
            else: # Linear
                ## ---- prune node ---- ##
                importance_c = np.abs(weight).sum((0))
                sorted_idx_c = np.argsort(importance_c)
                preserve_idx_c = sorted_idx_c[num_prune_c:] if num_prune_c != current_state['feature_channel'] else range(0, weight.shape[1])
                mask_c = np.ones(weight.shape[1], bool)
                mask_c[preserve_idx_c] = False
                weight[:,mask_c] = 0
                
                ## ---- update information ---- ##
                if num_prune_c == 0:
                    C_ratio = current_state['feature_channel']
                    FLOPs = 0
                    bias = 0
                else:
                    C_ratio = num_prune_c
                    FLOPs = current_state['FLOPs']
                    bias = current_state['kernel_out_channels']
                Reduced = current_state['Reduced'] + (C_ratio/current_state['feature_channel']) * (FLOPs - bias)
                previous_height = current_state['feature_height']
                previous_width = current_state['feature_width']
                x = 0
                y = 0
                receptive_height = current_state['feature_height']
                receptive_width = current_state['feature_width']
        
        elif self.method == 'rectangular':
            assert len(action) == 2, "Rectangular pruning action should be single float number"
            
            num_prune_h = int(round((1 - action[0]) * current_state['kernel_height']))
            num_prune_w = int(round((1 - action[1]) * current_state['kernel_width']))
            ## ---- execute action ---- ##
            if current_state['kernel_type'] == 0: # Convolutional
                ## ---- prune kernel ---- ##
                importance_h = np.abs(weight).sum((0,1,2))
                importance_w = np.abs(weight).sum((0,1,3))
                sorted_idx_h = np.argsort(importance_h)
                sorted_idx_w = np.argsort(importance_w)
                preserve_idx_h = sorted_idx_h[num_prune_h:] if num_prune_h != current_state['kernel_height'] else range(0, weight.shape[3])
                preserve_idx_w = sorted_idx_w[num_prune_w:] if num_prune_w != current_state['kernel_width'] else range(0, weight.shape[2])
                mask_h = np.ones(weight.shape[2], bool)
                mask_w = np.ones(weight.shape[3], bool)
                mask_h[preserve_idx_h] = False
                mask_w[preserve_idx_w] = False
                weight[:,:,mask_h,:] = 0
                weight[:,:,:,mask_w] = 0
                
                ## ---- update information ---- ##
                if num_prune_h == 0:
                    H_ratio = current_state['kernel_height']
                else:
                    H_ratio = num_prune_h
                if num_prune_w == 0:
                    W_ratio = current_state['kernel_width']
                else:
                    W_ratio = num_prune_w
                if num_prune_h == 0 and num_prune_w == 0:
                    FLOPs = 0
                else:
                    FLOPs = current_state['FLOPs']
                Reduced = current_state['Reduced'] + (H_ratio/current_state['kernel_height'])*(W_ratio/current_state['kernel_width'])*FLOPs
                previous_height = current_state['feature_height']
                previous_width = current_state['feature_width']
                x = max(0, (np.where(mask_h==False)[0][0] - current_state['padding_height'])) if num_prune_h != current_state['kernel_height'] else 0
                y = max(0, (np.where(mask_w==False)[0][0] - current_state['padding_width'])) if num_prune_w != current_state['kernel_width'] else 0
                receptive_height = min((current_state['feature_height']-x),(2*current_state['padding_height'] + current_state['feature_height'] - np.where(mask_h==False)[0][0] - len(mask_h) + np.where(mask_h==False)[0][-1] + 1)) if num_prune_h != current_state['kernel_height'] else current_state['feature_height']
                receptive_width = min((current_state['feature_width']-y),(2*current_state['padding_width'] + current_state['padding_width'] + current_state['feature_width'] - np.where(mask_w==False)[0][0] - len(mask_w) + np.where(mask_w==False)[0][-1] + 1)) if num_prune_w != current_state['kernel_width'] else current_state['feature_width']
                
            else:# Rectangular doesn't have way to prune linear kernel
                Reduced = current_state['Reduced']
                previous_height = current_state['feature_height']
                previous_width = current_state['feature_width']
                x = 0
                y = 0
                receptive_height = current_state['feature_height']
                receptive_width = current_state['feature_width']
            
        elif self.method == 'both':
            assert len(action) == 3, "Channel and rectangular pruning action should be pair of float numbers"
            
            num_prune_c = int(round((1 - action[0]) * current_state['feature_channel']))
            num_prune_h = int(round((1 - action[1]) * current_state['kernel_height']))
            num_prune_w = int(round((1 - action[2]) * current_state['kernel_width']))
            ## ---- execute action ---- ##
            if current_state['kernel_type'] == 0: # Convolutional
                ## ---- prune channel and kernel ---- ##
                importance_c = np.abs(weight).sum((0,2,3))
                importance_h = np.abs(weight).sum((0,1,2))
                importance_w = np.abs(weight).sum((0,1,3))
                sorted_idx_c = np.argsort(importance_c)
                sorted_idx_h = np.argsort(importance_h)
                sorted_idx_w = np.argsort(importance_w)
                preserve_idx_c = sorted_idx_c[num_prune_c:] if num_prune_c != 0 else range(0, weight.shape[1])
                preserve_idx_h = sorted_idx_h[num_prune_h:] if num_prune_h != 0 else range(0, weight.shape[3])
                preserve_idx_w = sorted_idx_w[num_prune_w:] if num_prune_w != 0 else range(0, weight.shape[2])
                mask_c = np.ones(weight.shape[1], bool)
                mask_h = np.ones(weight.shape[2], bool)
                mask_w = np.ones(weight.shape[3], bool)
                mask_c[preserve_idx_c] = False
                mask_h[preserve_idx_h] = False
                mask_w[preserve_idx_w] = False
                weight[:,mask_c,:,:] = 0
                weight[:,:,mask_h,:] = 0
                weight[:,:,:,mask_w] = 0
                
                ## ---- update information ---- ##
                if num_prune_c == 0:
                    C_ratio = current_state['feature_channel']
                else:
                    C_ratio = num_prune_c
                if num_prune_h == 0:
                    H_ratio = current_state['kernel_height']
                else:
                    H_ratio = num_prune_h
                if num_prune_w == 0:
                    W_ratio = current_state['kernel_width']
                else:
                    W_ratio = num_prune_w
                if num_prune_c == 0 and num_prune_h == 0 and num_prune_w == 0:
                    FLOPs = 0                 
                else:
                    FLOPs = current_state['FLOPs']
                Reduced = current_state['Reduced'] + (C_ratio/current_state['feature_channel'])*(H_ratio/current_state['kernel_height'])*(W_ratio/current_state['kernel_width'])*FLOPs
                previous_height = current_state['feature_height']
                previous_width = current_state['feature_width']
                x = max(0, (np.where(mask_h==False)[0][0] - current_state['padding_height'])) if num_prune_h != current_state['kernel_height'] else 0
                y = max(0, (np.where(mask_w==False)[0][0] - current_state['padding_width'])) if num_prune_w != current_state['kernel_width'] else 0
                receptive_height = min((current_state['feature_height']-x),(2*current_state['padding_height'] + current_state['feature_height'] - np.where(mask_h==False)[0][0] - len(mask_h) + np.where(mask_h==False)[0][-1] + 1)) if num_prune_h != current_state['kernel_height'] else current_state['feature_height']
                receptive_width = min((current_state['feature_width']-y), (2*current_state['padding_width'] + current_state['feature_width'] - np.where(mask_w==False)[0][0] - len(mask_w) + np.where(mask_w==False)[0][-1] + 1)) if num_prune_w != current_state['kernel_width'] else current_state['feature_width']
                
            else: # Linear
                ## ---- prune node ---- ##
                importance_c = np.abs(weight).sum((0))
                sorted_idx_c = np.argsort(importance_c)
                preserve_idx_c = sorted_idx_c[num_prune_c:] if num_prune_c != current_state['feature_channel'] else range(0, weight.shape[1])
                mask_c = np.ones(weight.shape[1], bool)
                mask_c[preserve_idx_c] = False
                weight[:,mask_c] = 0
                
                ## ---- update information ---- ##
                if num_prune_c == 0:
                    C_ratio = current_state['feature_channel']
                    FLOPs = 0
                    bias = 0
                else:
                    C_ratio = num_prune_c
                    FLOPs = current_state['FLOPs']
                    bias = current_state['kernel_out_channels']
                    
                Reduced = current_state['Reduced'] + (C_ratio/current_state['feature_channel']) * (FLOPs - bias)
                previous_height = current_state['feature_height']
                previous_width = current_state['feature_width']
                x = 0
                y = 0
                receptive_height = current_state['feature_height']
                receptive_width = current_state['feature_width']
        else:
            raise NameError("You should select proper method.")
        
        ## ---- assign pruned weight array to parameter tensor ---- ##
        current_kernel.weight.data = torch.from_numpy(weight).to(self.device)
        
        ## ---- revise next state information ---- ##
        self.current_state_idx += 1    
        
        next_state = self.episodic_states[self.current_state_idx]
        next_state['Reduced'] = Reduced
        next_state['previous_height'] = previous_height
        next_state['previous_width'] = previous_width
        next_state['x'] = x
        next_state['y'] = y
        next_state['receptive_height'] = receptive_height
        next_state['receptive_width'] = receptive_width
        
        ## ---- return next state ---- ##
        #print(next_state)
        #print("Step: %d/%d" % (self.current_state_idx+1,len(self.prunable_layer)+1))
        if self.current_state_idx == len(self.prunable_layer): # Final
            return next_state, True
        else:
            return next_state, False
        
    def get_reward(self, final_state, acc_standard, acc_least, flop_standard, flop_least):
        with torch.no_grad():
            criterion = nn.CrossEntropyLoss()
            self.model.eval()
            correct = 0
            loss = 0
            for x, y in self.validation_set:
                output = self.model(x.float().to(self.device))
                loss_tmp = criterion(output, y.long().to(self.device))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(self.device).view_as(pred)).sum().item()
                loss += loss_tmp.item()
        
        accuracy_pruned = correct/self.half_length #if min(correct/self.half_length,1) != 0.1 else 0
        FLOPs_pruned = final_state['Reduced']
        loss /= len(self.validation_set)
        FLOP_ratio = (FLOPs_pruned/self.total_FLOPs)
        
        #reward = (1-loss**0.4) * (max(FLOP_ratio,0.1))**(1/max(loss,0.1))
        #a = 1 - (1-accuracy_pruned)**0.4
        #b = (1 - max((1-FLOP_ratio),0.3))**(1/max((1-accuracy_pruned),0.2))
        #reward = a * b
        #reward = (accuracy_pruned/self.accuracy_original)*(FLOPs_pruned/self.total_FLOPs)
        
        #print(accuracy_pruned)
        #print(FLOP_ratio)
        
        if accuracy_pruned > acc_standard:
            acc_reward = ((1-acc_standard)/math.log(2-acc_standard))*math.log(accuracy_pruned+1-acc_standard)+acc_standard
        elif accuracy_pruned > acc_least and accuracy_pruned <= acc_standard:
            acc_reward = (-acc_standard/math.log(1-acc_least+acc_standard))*math.log(-accuracy_pruned+1+acc_standard)+acc_standard
        else:
            acc_reward = 0
        
        if FLOP_ratio > flop_standard:
            flop_reward = ((1-flop_standard)/math.log(2-flop_standard))*math.log(FLOP_ratio+1-flop_standard)+flop_standard
        elif FLOP_ratio > flop_least and FLOP_ratio <= flop_standard:
            flop_reward = (-flop_standard/math.log(1-flop_least+flop_standard))*math.log(-FLOP_ratio+1+flop_standard)+flop_standard
        else:
            flop_reward = 0
        
        reward = acc_reward * flop_reward
        
        return reward
    
    def test(self, final_state):
        criterion = nn.CrossEntropyLoss()
        accuracy = 0
        loss = 0
        with torch.no_grad():
            self.model.eval()
            correct = 0
            for x, y in self.test_set:
                output = self.model(x.float().to(self.device))
                loss_tmp = criterion(output, y.long().to(self.device))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(self.device).view_as(pred)).sum().item()
                loss += loss_tmp.item()
            accuracy = correct / self.half_length if min(correct/self.half_length,1) != 0.1 else 0
            loss /= len(self.test_set)
        
        FLOPs_pruned = final_state['Reduced']
        
        print("="*100)
        print("Pruned Model's Accuracy is %f" % accuracy)
        #print("Pruned Model's Loss is %f" % loss)
        print("Pruned Model's FLOPs Ratio is %f" % (FLOPs_pruned/self.total_FLOPs))
        
        reward = (accuracy/self.accuracy_original)*(FLOPs_pruned/self.total_FLOPs)
                
        print("Test Reward is %f" % reward)
    
    def original_result(self):
        _, _ = self.reset()
        criterion = nn.CrossEntropyLoss()
        accuracy = 0
        loss = 0
        with torch.no_grad():
            self.model.eval()
            correct = 0
            for x, y in self.test_set:
                output = self.model(x.float().to(self.device))
                loss_tmp = criterion(output, y.long().to(self.device))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(self.device).view_as(pred)).sum().item()
                loss += loss_tmp.item()
            accuracy = correct / self.half_length
            loss /= len(self.test_set)
        
        print("="*100)
        print("Original Model's Accuracy is %f" % accuracy)
        #print("Original Model's Loss is %f" % loss)
        
    def fine_tune(self, re_initialize=False):
        
        return


        
    

