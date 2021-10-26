import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
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

