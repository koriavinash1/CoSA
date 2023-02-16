from __future__ import print_function

import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normal_init(m, mean=0., std=0.01):
    m.weight.data.normal_(mean, std)


class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)

        return h



class Model(nn.Module):
    def __init__(self, number_components = 50, dim = 256):
        super(Model, self).__init__()
        self.dim = dim
        self.number_components = number_components

    # AUXILIARY METHODS
    def add_pseudoinputs(self):

        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)

        self.means = NonLinear(self.number_components, 
                                self.dim, 
                                bias=False, 
                                activation=nonlinearity)

        # init pseudo-inputs
        normal_init(self.means.linear, 0.05, 0.01)

        # create an idle input for calling pseudo-inputs
        self.idle_input = Variable(torch.eye(self.number_components, 
                                                self.number_components), 
                                            requires_grad=False)
        
        self.to(device)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(mu.shape).to(mu.device)
        return eps.mul(std).add_(mu)

    def calculate_loss(self):
        return 0.

    def calculate_likelihood(self):
        return 0.

    def calculate_lower_bound(self):
        return 0.

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        return 0.

#=======================================================================================================================