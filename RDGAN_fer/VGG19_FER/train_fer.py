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
#from models.VGG_gru import FERANet
#from models.VGG_model import FERANet2

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG16', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER_cycori', help='FER_RD_16_BCE / FER2013 / FER_cycori')
parser.add_argument('--bs', default=128, type=int, help='learning rate')
parser.add_argument('--ts', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dataroot', type=str ,default='FER2013', help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--layer', type=int ,default=1, help='learning rate')
opt = parser.parse_args()



f = Fer_DataLoader(opt.dataroot)
train_data = f.get_data(opt.bs)
test_data = f.get_test_data2(opt.ts)
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
#model = ResNet18()
#model = FERANet2()
model.cuda()

start_epoch = 0
total_epoch = 250
state =0 
best_PrivateTest_acc = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=opt.lr,momentum=0.9)#,  weight_decay = 5e-4)
#scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.1)

def train(self):
    print('\nEpoch: %d' % epoch)
    if epoch > 80 :
        frac = (epoch - 80) // 5
        decay_factor = 0.9 ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    train_loss = 0
    total_loss = 0
    total = 0
    correct = 0
    print('learning_rate: %s' % str(current_lr))

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
        #print(nn.Softmax(_))
        #print("1", output.data)
        #print(torch.max(output.data,1))
        total += train_y.size(0)
        correct += predicted.eq(train_y.data).cpu().sum().item()
    train_acc = 100.*correct/total
    print(train_acc)
    print("Train_acc : %0.3f" % train_acc)
#    if epoch == 100 :
#        torch.save(model.state_dict(),'./fer_train.t7')
    
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
        bs, c, h, w = np.shape(train_x)
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
        torch.save(model.state_dict(), './cycori_train.t7')
        
    print("current_Private_acc: %0.3f" % PrivateTest_acc)
    print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
    print("best_PrivateTest_epoch: " + str(best_PrivateTest_acc_epoch))
    if epoch == 119 :
        file = open("./results_txt/Resnet/"+opt.model+'/'+opt.dataroot+'_'+str(opt.lr)+'.txt','a')
        file.write("current_Private_acc: %0.3f" % PrivateTest_acc+"\n")
        file.write("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc+"\n")
        file.write("best_PrivateTest_epoch: " + str(best_PrivateTest_acc_epoch)+'\n')
        file.close()

for epoch in range(120):
    train(epoch)
    PrivateTest(epoch)