"""
Name: model.py
Author: Xuewen Zhang
Date: 18/04/2024
Project: DeepDeePC
"""

import torch
import torch.nn as nn
import numpy as np

from .MyTool import MyTool, NNtool, timer


class network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size_list=[150, 150, 150]):
        super(network, self).__init__()
        """
           Args:
               args:
               num_classes: output classes num
               input_size: input model size
               hidden_size_list: the used history samples size in NN model (length=3)
               batch_size: the batch size
       """
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size_list[0])
        self.fc2 = nn.Linear(in_features=hidden_size_list[0], out_features=hidden_size_list[1])
        self.fc3 = nn.Linear(in_features=hidden_size_list[1], out_features=hidden_size_list[2])
        self.fc4 = nn.Linear(in_features=hidden_size_list[2], out_features=output_size)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()



    def forward(self, x):
        """
            x: (batch_size, x_dim/input_size)
            output x5: (batch_size, output_size)
        """
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)
        if torch.cuda.is_available():
            x = x.to(torch.device("cuda:0"))

        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.fc2(x2)
        x4 = self.relu(x3)
        x5 = self.fc3(x4)
        x6 = self.relu(x5)
        x7 = self.fc4(x6)
        # x6 = self.dropout(x5)
        return x7


    def loss(self, x, x_label):
        """
           x: predicted x states
           x_label: real x states
        """
        if not torch.is_tensor(x_label):
            x_label = torch.FloatTensor(x_label)
        if torch.cuda.is_available():
            x_label = x_label.to(torch.device("cuda:0"))

        mseloss = nn.MSELoss()

        # Define Loss of X lstm model
        x_loss = mseloss(x_label, x)

        return x_loss
    
    



            