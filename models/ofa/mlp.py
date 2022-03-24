# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, ffn_layernorm, dropout_module, activation_dropout_module):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.ffn_layernorm = ffn_layernorm
        self.dropout_module = dropout_module
        self.activation_dropout_module = activation_dropout_module

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout_module(out)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        out = self.fc2(out)
        out = self.dropout_module(out)
        return out
