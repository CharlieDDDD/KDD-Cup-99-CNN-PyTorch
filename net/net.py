# -*- coding:utf-8 -*-
#@Time  : 2020/6/6 15:16
#@Author: DongBinzhi
#@File  : net.py

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """(Linear => ReLU) => (Linear => [dropout] => ReLU) * 2 => (Linear)"""
    
    def __init__(self, input_dim, hidden_1, hidden_2, hidden_3, out_dim):
        super(Net, self).__init__()
       # self.fc1 = nn.Sequential(nn.Linear(input_dim,hidden_1), nn.BatchNorm1d(hidden_1), nn.ReLU(inplace=True),)
       # self.fc2 = nn.Sequential(nn.Linear(hidden_1,hidden_2), nn.BatchNorm1d(hidden_2), nn.Dropout(0.5), nn.ReLU(inplace=True))
       # self.fc3 = nn.Sequential(nn.Linear(hidden_2,hidden_3), nn.BatchNorm1d(hidden_3), nn.Dropout(0.5), nn.ReLU(inplace=True))
       # self.fc4 = nn.Sequential(nn.Linear(hidden_3,out_dim))
        
        self.fc1 = nn.Sequential(nn.Linear(input_dim,hidden_1), nn.ReLU(inplace=True),)
        self.fc2 = nn.Sequential(nn.Linear(hidden_1,hidden_2), nn.Dropout(0.5), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(hidden_2,hidden_3),  nn.Dropout(0.5), nn.ReLU(inplace=True))
        self.fc4 = nn.Sequential(nn.Linear(hidden_3,out_dim))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x




