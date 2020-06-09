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
        self.c1  = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3,padding=1), 
                                #nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True))
        self.c2  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3,padding=1), 
                                #nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        #self.c3  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3,padding=1), 
                                #nn.BatchNorm2d(64),
                                #nn.ReLU(inplace=True))                                
        self.p1  = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(64*3*3,hidden_1), nn.Dropout(0.5),nn.ReLU(inplace=True))
        #self.fc2 = nn.Sequential(nn.Linear(hidden_1,hidden_3), nn.Dropout(0.5), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(hidden_1,hidden_3),  nn.Dropout(0.5), nn.ReLU(inplace=True))
        self.fc4 = nn.Sequential(nn.Linear(hidden_3,out_dim))

    def forward(self, x):
        x = self.c1(x)
       # print('x.shape',x.shape)
        x = self.c2(x)
        
       # x =self.c3(x)
       # print('x.shape',x.shape)
        x = self.p1(x)
        
        #print('x.shape',x.shape)
        x = x.view(x.size(0), -1)
        
       # print('x.shape',x.shape)
        x = self.fc1(x)
        
        #print('x.shape',x.shape)
        #x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x




