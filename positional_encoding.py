import torch
import torch.nn as nn
import math

# Source: https://medium.com/towards-data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        """
        Positional Encoding
        :param d_model: Integer, dimension of the encoding
        :param max_seq_length:  Integer, maximum length of a sentence
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) # 10000  mentioned in the paper

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

