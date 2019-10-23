import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo


class FERANet(nn.Module):
    def __init__(self):
        super(FERANet, self).__init__()
        
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.batn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.batn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.batn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.batn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.batn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.batn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.batn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.batn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.batn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.batn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.batn4_4 = nn.BatchNorm2d(512)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.batn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.batn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.batn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.batn5_4 = nn.BatchNorm2d(512)
        self.relu5_4 = nn.ReLU(inplace=True)

        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Linear(512*2*2, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096 , 512)
        #self.gru = nn.GRU(4096,128, batch_first=True)

        self.classify = nn.Linear(512, 7)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):

        h = x
        h = self.relu1_1(self.batn1_1((self.conv1_1(h))))
        h = self.relu1_2(self.batn1_2((self.conv1_2(h))))
        h = self.pool1(h)

        h = self.relu2_1(self.batn2_1((self.conv2_1(h))))
        h = self.relu2_2(self.batn2_2((self.conv2_2(h))))
        h = self.pool2(h)
        
        h = self.relu3_1(self.batn3_1((self.conv3_1(h))))
        h = self.relu3_2(self.batn3_2((self.conv3_2(h))))
        h = self.relu3_3(self.batn3_3((self.conv3_3(h))))
        h = self.pool3(h)
        
        h = self.relu4_1(self.batn4_1((self.conv4_1(h))))
        h = self.relu4_2(self.batn4_2((self.conv4_2(h))))
        h = self.relu4_3(self.batn4_3((self.conv4_3(h))))
        h = self.relu4_4(self.batn4_4((self.conv4_4(h))))
        h = self.pool4(h)
        
        h = self.relu5_1(self.batn5_1((self.conv5_1(h))))
        h = self.relu5_2(self.batn5_2((self.conv5_2(h))))
        h = self.relu5_3(self.batn5_3((self.conv5_3(h))))
        h = self.relu5_4(self.batn5_4((self.conv5_4(h))))
        h = self.pool5(h)
        
        h = h.view(-1, self.num_flat_features(h))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        x = F.relu(self.fc8(h))
        x = x.view(x.size(0), -1, self.num_flat_features(x))   #1,16,4096 batchsize,sequence_length,data_dim

        #x, hn = self.gru(x)

        x = self.dropout(x)

        x = torch.mean(x,1)

        x = self.classify(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]   
        num_features = 1
        for s in size:
            num_features *= s
        return num_features









