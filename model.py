# -*- coding: utf-8 -*-
# @Time : 2023/3/1 9:46
# @Author : Wanghong
# @FileName: model.py
# @Software: PyCharm
import copy
import os
import torch.utils.data
from torch import nn, FloatTensor
from torch.nn.functional import relu
from torch.nn.utils.rnn import *
# from const import *
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Parameter
import torchvision.models as models
from torch.nn import functional as F
from torchsummary import summary


class RNN(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, max_length, n_input=30, n_output=1, n_hidden=128, n_rnn_layer=2, bidirectional=True, dropout=0.5):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(n_input, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True)

        self.linear1 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1) * max_length, 256)
        self.linear2 = torch.nn.Linear(256, 64)
        self.linear3 = torch.nn.Linear(64, n_output)

        self.dropout = torch.nn.Dropout(dropout)
        self.num_layers = n_rnn_layer
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = n_hidden

    def forward(self, x, lens):
        batch = x.shape[0]
        length = x.shape[1]

        h0 = torch.randn(self.num_layers * self.num_directions, batch, self.hidden_size).to(x.device)
        c0 = torch.randn(self.num_layers * self.num_directions, batch, self.hidden_size).to(x.device)

        sequence = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        inputs, _ = self.rnn(sequence, (h0, c0))
        inputs, _ = pad_packed_sequence(inputs, batch_first=True, total_length=length)

        x = relu(self.linear1(inputs.view(batch, -1)))
        x = relu(self.linear2(self.dropout(x)))
        y1 = self.linear3(x)
        return y1.view(batch)


class Sharifi_CNN(nn.Module):
    def __init__(self, input_length=152):
        super(Sharifi_CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=30, out_channels=32, kernel_size=30)
        self.bn1 = nn.BatchNorm1d(32)
        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=15)
        self.bn2 = nn.BatchNorm1d(16)
        # Compute the size of the input to the first fully connected layer
        conv1_output_length = (input_length - 30 + 1) // 2
        conv2_output_length = (conv1_output_length - 15 + 1) // 2
        fc1_input_size = 16 * conv2_output_length
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=fc1_input_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # translate dimen1 and dimen2
        x = x.permute(0, 2, 1)
        batch = x.shape[0]
        # Apply first convolutional layer with ReLU activation and max pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        # Apply second convolutional layer with ReLU activation and max pooling
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        # Flatten the output from convolutional layers
        x = x.view(x.size(0), -1)
        # Apply first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply second fully connected layer to get the stride length output
        x = self.fc2(x)
        return x.view(batch)

