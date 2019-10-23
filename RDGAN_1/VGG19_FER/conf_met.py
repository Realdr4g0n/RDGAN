"""
plot confusion_matrix of PublicTest and PrivateTest
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse

from torch.autograd import Variable
import torchvision
import transforms as transforms
from sklearn.metrics import confusion_matrix
from models import *

from Fer_DataLoader import Fer_DataLoader

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='FER_RD_16_BCE / FER2013')
parser.add_argument('--bs', default=128, type=int, help='learning rate')
parser.add_argument('--ts', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--layer', type=int ,default=1, help='learning rate')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()


class_names = ['Angry','Disgust','Fear','Happy','Neutral','sad','surprised']
opt = parser.parse_args()

f = Fer_DataLoader(opt.dataset)

test_data = f.get_test_data2(128)

if opt.model == 'VGG19':
    net = VGG('VGG16',1)
elif opt.model  == 'Resnet18':
    net = ResNet18()
net.cuda()

net.load_state_dict(torch.load('./ori_train.t7'))###수정해야함
net.eval()

correct = 0
total =0

all_target=[]

for batch_ids,(test_x,test_y) in enumerate(test_data) :
    test_x,test_y = test_x.cuda() , test_y.cuda()
    test_x,test_y = Variable(test_x),Variable(test_y)
    #bs,c,h,w = np.shape(test_x)
    output = net(test_x)
    
    _, predicted = torch.max(output.data,1)
    total += test_y.size(0)
    correct += predicted.eq(test_y.data).cpu().sum().item()
    
    if batch_ids == 0:
        all_predicted = predicted
        all_targets = test_y
    
    else :
        all_predicted = torch.cat((all_predicted, predicted),0)
        all_targets = torch.cat((all_targets, test_y),0)
    
acc = 100. * correct / total
print ("accuracy: %0.3f" % acc)

matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
np.set_printoptions(precision=2)

plt.figure(figsize=(10,9))
plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                      title= '1 Confusion Matrix (Accuracy: %0.3f%%)' %acc)
plt.savefig(os.path.join('./1_cm.png'))
plt.close()