import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection

#fer_dataset = './Datasets/FER_RD_16_BCE/'
#fer_dataset = './Datasets/FER2013/'
batch_size = 128
test_batch_size = 128
class Fer_DataLoader():
    def __init__(self, Dataroot):
        super(Fer_DataLoader, self).__init__()
        self.fer_dataset = './Datasets/' + Dataroot  
    
    def get_data(self,batch_size2):
        dirs = ['angry','disgusted','fearful','happy','neutral','sad','surprised']

        data = []
        label = []
        for i, d in enumerate(dirs):

            files = os.listdir('./'+ self.fer_dataset+'/train/' + d)

            for f in files:
                img = Image.open('./'+ self.fer_dataset +'/train/' + d +'/' + f, 'r')

                #resize = img.resize((x,y))
                #img = img.resize((256,256),PIL.Image.BICUBIC) # 할 필요가 없을꺼 같은데?

                resized_img = np.asarray(img)
                resized_img = np.asarray([resized_img])
                if str(resized_img.shape) == '(1,48,48,3)':
                    print(str(img.filename))
                
                data.append(resized_img)

                label.append(i)
                img.close()

        data = np.array(data, dtype='float32')
        label = np.array(label, dtype='int64')
        
        train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data,label,test_size=0.1)
        train_X = torch.from_numpy(train_X).float()
        train_Y = torch.from_numpy(train_Y).long()
        train = TensorDataset(train_X,train_Y)
        train_loader = DataLoader(train, batch_size = batch_size2, shuffle=True)
        return train_loader

    def get_test_data2(self,test_size):
        dirs = ['angry','disgusted','fearful','happy','neutral','sad','surprised']
        
        data = []
        label = []
        for i, d in enumerate(dirs):
            files = os.listdir('./'+ self.fer_dataset +'/test/' + d)

            for f in files:
                img = Image.open('./'+ self.fer_dataset +'/test/' + d +'/' + f, 'r')

                resized_img = np.asarray(img)
                resized_img = np.asarray([resized_img])

                data.append(resized_img)

                label.append(i)
                img.close()

        data = np.array(data, dtype='float32')
        label = np.array(label, dtype='int64')
        
        train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data,label,test_size=0.1)
        train_X = torch.from_numpy(train_X).float()
        train_Y = torch.from_numpy(train_Y).long()
        train = TensorDataset(train_X,train_Y)
        test_loader = DataLoader(train, batch_size = test_size, shuffle=True)
    
        return test_loader
    
    def get_test_data(self):
        dirs = ['angry','disgusted','fearful','happy','neutral','sad','surprised']

        data = []
        label = []
        for i, d in enumerate(dirs):

            files = os.listdir('../datasets/D2N/trainB/')

            for f in files:
                img = Image.open('../datasets/D2N/trainB' + '/' + f, 'r')

                #resize = img.resize((x,y))

                resized_img = np.asarray(img)
                resized_img = np.asarray([resized_img])

                data.append(resized_img)

                label.append(i)
                img.close()

        data = np.array(data, dtype='float32')
        label = np.array(label, dtype='int64')
        
        train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data,label,test_size=0.1)
        train_X = torch.from_numpy(train_X).float()
        train_Y = torch.from_numpy(train_Y).long()
        train = TensorDataset(train_X,train_Y)
        train_loader = DataLoader(train, batch_size = 1, shuffle=True)
        return train_loader