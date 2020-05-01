# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:57:41 2020

@author: SFC202004009
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from utils import read_ZINC_smiles, smiles_to_onehot
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import sys
import argparse
import time

class DataSet(Dataset):
    def __init__(self, smi_list, logP_list):
        self.smi_list = smi_list
        self.logP_list = logP_list
        
    def __len__(self):
        return len(self.logP_list)
    
    def __getitem__(self, index):
        return self.smi_list[index], self.logP_list[index]
    
    
def make_partition():
    smi_list, logP_total, tpsa_total = read_ZINC_smiles(50000)
    smi_total = smiles_to_onehot(smi_list)
    
    num_train = 30000
    num_validation = 10000
    num_test = 10000
    
    smi_train = smi_total[0:num_train]
    logP_train = logP_total[0:num_train]
    smi_validation = smi_total[num_train:(num_train+num_validation)]
    logP_validation = logP_total[num_train:(num_train+num_validation)]
    smi_test = smi_total[(num_train+num_validation):]
    logP_test = logP_total[(num_train+num_validation):]
    
    train_dataset = DataSet(smi_train, logP_train)
    val_dataset = DataSet(smi_validation, logP_validation)
    test_dataset = DataSet(smi_test, logP_test)
    
    partition = {'train': train_dataset,
                 'val': val_dataset,
                 'test': test_dataset}
    
    return partition

class Cnn(nn.Module):
    def __init__(self, num_layer, hidden_dim):
        super(Cnn, self).__init__()
        
        self.n_layer = num_layer
        self.convs = nn.ModuleList()
        
        c = nn.Conv1d(31, 64, 9, padding = 4, bias = True)
        self.convs.append(c)
        for i in range(1, num_layer):
            c = nn.Conv1d(64, 64, 9, padding = 4, bias = True)
            self.convs.append(c)
        
        self.l1 = nn.Linear(7680, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.xavier_init()
        
    def forward(self, x):
        for conv in self.convs:
            x = self.relu(conv(x))
        x1 = x
        x = x.view(-1, x.shape[1] * x.shape[2])
        
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.tanh(x)
        x = self.l3(x)
        
        return x, x1
        
    def make_conv(self,in_channels, out_channels, kernel_size):
        return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size, padding = 4, bias = True),
                             nn.ReLU())
    
    def xavier_init(self):
        for conv in self.convs:
            nn.init.xavier_normal_(conv.weight)
            conv.bias.data.fill_(0.01)
        
        nn.init.xavier_normal_(self.l1.weight)
        self.l1.bias.data.fill_(0.01)
        
        nn.init.xavier_normal_(self.l2.weight)
        self.l2.bias.data.fill_(0.01)
        
        nn.init.xavier_normal_(self.l3.weight)
        self.l3.bias.data.fill_(0.01)
            

def make_one_hot(nump, num):
    one = np.identity(31, dtype = np.float)
    nump_one_hot = np.zeros((nump.shape[0], nump.shape[1], num), dtype = np.float)
    for i_n, i in enumerate(nump):
        for j_n, j in enumerate(i):
            nump_one_hot[i_n, j_n] = one[j]
    return nump_one_hot

def train(net, partition, optimizer, criterion, args):
    trainloader = DataLoader(partition['train'], batch_size = args.train_batch, shuffle = True, num_workers = 0)
    
    net.train()
    optimizer.zero_grad()

    total = 0
    train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        nump = inputs.numpy()
        inputs_one_hot = make_one_hot(nump, 31)
        inputs = torch.from_numpy(inputs_one_hot)
        inputs = inputs.permute(0, 2, 1)
        inputs = inputs.float()
        labels = labels.float()
        
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs, _o = net(inputs)
        
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        total += labels.size(0)
        
    train_loss = train_loss / len(trainloader)
    return net, train_loss

def validate(net, partition, criterion, args):
    trainloader = DataLoader(partition['val'], batch_size = args.val_batch, shuffle = True, num_workers = 0)
    
    net.eval()
    
    with torch.no_grad():
        total = 0
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            nump = inputs.numpy()
            inputs_one_hot = make_one_hot(nump,31)
            inputs = torch.from_numpy(inputs_one_hot)
            inputs = inputs.permute(0, 2, 1)
            inputs = inputs.float()
            labels = labels.float()
            
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs, _o = net(inputs)
            
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            
            train_loss += loss.item()
            total += labels.size(0)
        
        train_loss = train_loss / len(trainloader)
    
    return train_loss

def test(net, partition, criterion, args):
    trainloader = DataLoader(partition['test'], batch_size = args.test_batch, shuffle = True, num_workers = 0)
    
    net.eval()
    
    with torch.no_grad():
        total = 0
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            nump = inputs.numpy()
            inputs_one_hot = make_one_hot(nump,31)
            inputs = torch.from_numpy(inputs_one_hot)
            inputs = inputs.permute(0, 2, 1)
            inputs = inputs.float()
            labels = labels.float()
            
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs, _o = net(inputs)
            
            outputs= outputs.squeeze()
            loss = criterion(outputs, labels)
            
            train_loss += loss.item()
            total += labels.size(0)
            
            break
    labels = labels.cpu()
    outputs = outputs.cpu()
    plt.figure()
    plt.scatter(labels, outputs, s=3)
    plt.xlabel('logP - Truth', fontsize=15)
    plt.ylabel('logP - Prediction', fontsize=15)
    x = np.arange(-4,6)
    plt.plot(x,x,c='black')
    plt.tight_layout()
    plt.axis('equal')
    plt.show()

    return train_loss

def experiment(args):
    net = Cnn(args.num_layer, args.layer_size)
    partition = make_partition()
    net.cuda() 
    
    criterion = nn.MSELoss()

    for i in range(1, args.epoch):
        learning_rate = args.initial_lr * 0.95 ** i
        optimizer = optim.SGD(net.parameters(), lr = learning_rate, weight_decay = 0.00001)
        tic = time.time()
        net, train_loss = train(net, partition, optimizer, criterion, args)
        val_loss = validate(net, partition, criterion, args)
        
        tok = time.time()
        print('Epoch {}, Loss(train/val) {:2.3f}/{:2.3f} time {:2.2f} sec.'.format(i, train_loss, val_loss, tok-tic))
        
    test_loss = test(net, partition, criterion, args)
    print("test Loss: {}".format(test_loss))
    
    return

seed = 597101285227200
seed2 = 5457684948745992
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed2)

print('torch seed: ', torch.initial_seed())
print('torch cuda seed: ', torch.cuda.initial_seed())

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.epoch = 35
args.train_batch = 1000
args.val_batch = 1000
args.test_batch = 10000
args.initial_lr = 0.01
args.num_layer = 3
args.layer_size = 1024

experiment(args)