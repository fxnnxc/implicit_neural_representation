# -------------------------
# 2021.11.23 Bumjin Park 
# -------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from collections import OrderedDict

class TestModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.input_size = model_config['input_size']
        self.hidden_size = model_config['hidden_size']
        self.output_size = model_config['output_size']

        net = [] 
        prev_size = self.input_size
        for size in self.hidden_size:
            net.append(nn.Linear(prev_size, size))
            net.append(nn.ReLU())
            prev_size = size

        net.append(nn.Linear(prev_size, self.output_size))
        self.net = nn.Sequential(*net)

    def forward(self, input):
        output = self.net(input)
        return output 


# ------
# https://github.com/WANG-KX/SIREN-2D/blob/master/models.py

class ReLU_Model(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_features))
        for i in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
        self.layers.append(nn.Linear(hidden_features, out_features))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, get_gradient=False):
        # save middle result for gradient calculation
        x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        relu_masks = []
        middle_result = x 
        for layer in self.layers[:-1]:
            middle_result = self.relu(layer(middle_result))
            relu_mask = (middle_result > 0)
            relu_mask.type_as(middle_result)
            relu_masks.append(relu_mask)
        # last layer
        result = self.layers[-1](middle_result)

        if not get_gradient:
            return result, x

        # do backwards 
        B = x.shape[0]
        gradient = self.layers[-1].weight
        gradient = gradient.repeat(B,1)
        for i in range(len(self.layers)-2, -1, -1):
            layer_relu_mask = relu_masks[i]
            layer_gradient_weight = self.layers[i].weight
            gradient = gradient * layer_relu_mask
            gradient = torch.matmul(gradient, layer_gradient_weight)

        return result, x, gradient 


class ReLU_PE_Model(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, L):
        '''L is the level of position encoding'''
        super().__init__()
        self.net = nn.ModuleList()
        in_features = in_features + in_features * 2 * L
        self.net.append(nn.Linear(in_features, hidden_features))
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
        self.net.append(nn.Linear(hidden_features, out_features))
        self.relu = nn.ReLU(inplace=True)
        self.L = L

    def position_encoding_forward(self,x):
        B,C = x.shape
        x = x.view(B,C,1)
        results = [x]
        for i in range(1, self.L+1):
            freq = (2**i) * np.pi
            cos_x = torch.cos(freq*x)
            sin_x = torch.sin(freq*x)
            results.append(cos_x)
            results.append(sin_x)
        results = torch.cat(results, dim=2)
        results = results.permute(0,2,1)
        results = results.reshape(B,-1)
        return results

    def position_encoding_backward(self,x):
        B,C = x.shape
        x = x.view(B,C,1)
        results = [torch.ones_like(x)]
        for i in range(1, self.L+1):
            freq = (2**i) * np.pi
            cos_x_grad = -1.0*torch.sin(freq*x)*freq
            sin_x_grad = torch.cos(freq*x)*freq
            results.append(cos_x_grad)
            results.append(sin_x_grad)
        results = torch.cat(results, dim=2)
        results = results.permute(0,2,1)
        results = results.reshape(B,-1)
        return results

    def forward(self, x, get_gradient=False):
        # save middle result for gradient calculation
        x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        x = x.view(-1, 2)
        relu_masks = []
        x_pe = self.position_encoding_forward(x)
        middle_result = x_pe
        for layer in self.net[:-1]:
            middle_result = self.relu(layer(middle_result))
            relu_mask = (middle_result > 0)
            relu_mask.type_as(middle_result)
            relu_masks.append(relu_mask)
        # last layer
        result = self.net[-1](middle_result)

        if not get_gradient:
            return result, x

        # do backwards 
        B,C = x.shape
        gradient = self.net[-1].weight
        gradient = gradient.repeat(B,1)
        for i in range(len(self.net)-2, -1, -1):
            layer_relu_mask = relu_masks[i]
            layer_gradient_weight = self.net[i].weight
            gradient = gradient * layer_relu_mask
            gradient = torch.matmul(gradient, layer_gradient_weight)
        # backward the gradient of position encoding
        pe_gradient = self.position_encoding_backward(x)
        gradient = gradient * pe_gradient
        gradient = gradient.view(B, -1, C)
        gradient = torch.sum(gradient, dim=1, keepdim=False)
        return result, x, gradient



class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

class OffSetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset_bias = nn.Linear(1,1)
    
    def forward(self, x):
        return self.offset_bias(x)


class EoREN():
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):

        self.siren = Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear, first_omega_0, hidden_omega_0)
        self.offset = OffSetModel()
        

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        g = self.siren.net(coords)

        g2 = g.clone().detach().requires_grad_(False)
        h = self.offset(g2) 
        return h, g, coords   
    
