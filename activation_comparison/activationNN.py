import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time


class ActivationLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.activation = activation 

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      

    def forward(self, input):
        if self.activation:
            return self.activation(self.linear(input))
        else:
            return self.linear(input)
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        if self.activation:
            intermediate = self.linear(input)
            return self.activation(intermediate), intermediate
        else:
            v = self.linear(input)
            return v,v
        
    
    
class ActivationNN(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, activation=F.relu):
        super().__init__()
        self.activation = activation
        
        self.net = []
        self.net.append(ActivationLayer(in_features, hidden_features, activation=activation))

        for i in range(hidden_layers):
            self.net.append(ActivationLayer(hidden_features, hidden_features, activation=activation))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(1 / hidden_features) , 
                                              np.sqrt(1 / hidden_features) )
                
            self.net.append(final_linear)
        else:
            self.net.append(ActivationLayer(hidden_features, out_features, activation=activation))
        
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
            if isinstance(layer, ActivationLayer):
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