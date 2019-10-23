import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torch.utils.data import DataLoader, TensorDataset

import argparse
import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection
from Fer_DataLoader import Fer_DataLoader
#from models import VGG
from models import *

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG16', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='RAF_single_ori', help='CNN architecture')
parser.add_argument('--bs', default=64, type=int, help='learning rate')
parser.add_argument('--ts', default=64, type=int, help='learning rate')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--dataroot', type=str ,default='RAF_single_ori', help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--layer', type=int ,default=2, help='learning rate')
opt = parser.parse_args()

if opt.model == 'VGG16':
    model = VGG('VGG16',opt.layer)
elif opt.model == 'VGG19':
    model = VGG('VGG19',opt.layer)
elif opt.model  == 'Resnet18_0':
    model = ResNet18_0()
elif opt.model  == 'Resnet101_0':
    model = ResNet101_0()
elif opt.model  == 'Resnet18_1':
    model = ResNet18_1()
elif opt.model  == 'Resnet101_1':
    model = ResNet101_1()

f = Fer_DataLoader(opt.dataroot)
train_data = f.get_data(opt.bs)
test_data = f.get_test_data2(opt.ts)

#model = VGG('VGG19')
#model = ResNet18()
model = model.cuda()

start_epoch = 0
total_epoch = 250
state =0
best_PrivateTest_acc = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay = 5e-4)

def train(self):
    
    print('\nEpoch: %d' % epoch)
    if epoch > 80 :
        frac = (epoch - 80) // 5
        decay_factor = 0.9 ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
        
    print('learning_rate: %s' % str(current_lr))
    train_loss = 0
    total_loss = 0
    total = 0
    correct = 0
    for train_x, train_y in train_data:
        train_x, train_y = train_x.cuda(), train_y.cuda()
        train_x,train_y = Variable(train_x), Variable(train_y)
        optimizer.zero_grad()

        output = model(train_x)
        loss = criterion(output,train_y)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        
        train_loss += loss.data[0]
        
        _, predicted = torch.max(output.data,1)
        
        total += train_y.size(0)
        correct += predicted.eq(train_y.data).cpu().sum().item()
        
    train_acc = 100*correct/total
    print(train_acc)
    print("Train_acc : %0.3f" % train_acc)
    if int(train_acc) == 2000 :
        torch.save(model.state_dict(),'./raf_train.t7')
def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    model.eval()
    
    PrivateTest_loss = 0
    correct = 0
    total = 0
    
    for train_x,train_y in test_data:
        train_x, train_y = train_x.cuda(), train_y.cuda()
        train_x,train_y = Variable(train_x), Variable(train_y)
        #bs, c, h, w = np.shape(train_x)
        outputs = model(train_x)
        #outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs, train_y)
        PrivateTest_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += train_y.size(0)
        correct += predicted.eq(train_y.data).cpu().sum().item()
    # Save checkpoint.
    PrivateTest_acc = 100.*correct/total
    
    if PrivateTest_acc > best_PrivateTest_acc:
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch
    print("current_Private_acc: ", PrivateTest_acc)
    print("best_PrivateTest_acc: ", best_PrivateTest_acc)
    print("best_PrivateTest_epoch: " , best_PrivateTest_acc_epoch)
    if epoch == 119 :
        file = open("./result_txt/VGG/"+opt.model+'_'+opt.dataroot+'_'+str(opt.lr)+'_'+str(opt.layer)+'.txt','a')
        file.write("current_Private_acc: %0.3f" % PrivateTest_acc+"\n")
        file.write("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc+"\n")
        file.write("best_PrivateTest_epoch: " + str(best_PrivateTest_acc_epoch)+'\n')
        file.close()

for epoch in range(250):
    train(epoch)
    PrivateTest(epoch)