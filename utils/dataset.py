# -*- coding:utf-8 -*-
#@Time  : 2020/6/6 15:16
#@Author: DongBinzhi
#@File  : dataset.py


from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import csv, os, random
import torch.nn as nn


class BasicDataset(Dataset):
    """create the BasiceDataset"""
    
    def __init__(self, data_path = 'dataset/train_data.csv', train_flag = True):

        super(BasicDataset, self).__init__()
        self.train_flag = train_flag
        self.data_path = data_path
        print('the data path: ',self.data_path)
        #self.data_length = 0
        #the ignored columns:9-21
        self.ignored_col = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18,19,20,21]
        self.data, self.data_length =self.preprocess(data_path = self.data_path, train_flag = self.train_flag)
        self.data = np.delete(self.data, self.ignored_col, axis=1)


    def __len__(self):
        return self.data_length

    #@classmethod
   # def normalization(self, minibatch):

# return res
    
    #read the data
    #@classmethod
    def preprocess(cls, data_path, train_flag):
        csv_reader = csv.reader(open(data_path))
        datasets = []
        if train_flag:
            label0_data = []
            label1_data = []
            label2_data = []
            label3_data = []
            label4_data = []
            label_status = {}
            for row in csv_reader:
                data = []
                for char in row:
                    if char=='None':
                        data.append(0)
                    else:
                        data.append(np.float32(char))       # transform data from format of string to float32
                if data[-1] == 0:
                    label0_data.append(data)
                if data[-1] == 1:
                    label1_data.append(data)
                if data[-1] == 2:
                    label2_data.append(data)
                if data[-1] == 3:
                    label3_data.append(data)
                if data[-1] == 4:
                    label4_data.append(data)
                # record the number of different labels
                if label_status.get(str(int(data[-1])),0)>0:
                    label_status[str(int(data[-1]))] += 1
                else:
                    label_status[str(int(data[-1]))] = 1

            while len(label0_data) < 10000:
                label0_data = label0_data + label0_data
            label0_data = random.sample(label0_data, 10000)
            while len(label1_data) < 10000:
                label1_data = label1_data + label1_data
            label1_data = random.sample(label1_data, 10000)
            while len(label2_data) < 10000:
                label2_data = label2_data + label2_data
            label2_data = random.sample(label2_data, 10000)
            while len(label3_data) < 10000:
                label3_data = label3_data + label3_data
            label3_data = random.sample(label3_data, 10000)
            while len(label4_data) < 10000:
                label4_data = label4_data + label4_data
            label4_data = random.sample(label4_data, 10000)
        
            datasets = label0_data+label1_data+label2_data+label3_data+label4_data
        else:
            for row in csv_reader:
                data = []
                for char in row:
                    if char=='None':
                        data.append(0)
                    else:
                        data.append(np.float32(char))       # transform data from format of string to float32
                datasets.append(data)

        data_length = len(datasets)
        #datasets = self.normalization(datasets)
        minibatch = datasets
        data = np.delete(minibatch, -1, axis=1)
        labels = np.array(minibatch,dtype=np.int32)[:, -1]
        mmax = np.max(data, axis=0)
        mmin = np.min(data, axis=0)
        for i in range(len(mmax)):
            if mmax[i] == mmin[i]:
                mmax[i] += 0.000001     # avoid getting devided by 0
        res = (data - mmin) / (mmax - mmin)
        res = np.c_[res,labels]
        np.random.shuffle(datasets)
        print('init data completed!')
        return datasets, data_length
    def __getitem__(self, i):
        data  = np.array(np.delete(self.data[i], -1))
        label = np.array(self.data[i],dtype=np.int32)[-1]
        #return {'data': torch.from_numpy(data), 'label':torch.from_numpy(label)}
        return {'data': torch.from_numpy(data), 'label':label}



