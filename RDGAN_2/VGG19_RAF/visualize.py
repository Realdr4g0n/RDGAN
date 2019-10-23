"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
from Fer_DataLoader import Fer_DataLoader
a= 0

f= Fer_DataLoader()
train_data = f.get_test_data()
total = 0
for inputs,_ in train_data:
    cut_size = 44
    if total == 200 :
        break
    transform_test = transforms.Compose([
        transforms.TenCrop(48),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])
    
    
    '''
    raw_img = io.imread(i)
    #gray = rgb2gray(raw_img)
    gray = resize(raw_img, (48,48), mode='symmetric').astype(np.uint8)

    img = gray

    img = Image.fromarray(img)
    inputs = transform_test(img)
    '''
    class_names = ['Angry','disgusted','fearful','happy','neutral','sad','surprised']

    net = VGG('VGG19')
    net.load_state_dict(torch.load('../fer_train.t7'))
    net.cuda()
    net.eval()
    '''
    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    print("1 : " )
    print(inputs)
    
    print("2 : " )
    print(inputs)
    inputs = Variable(inputs, volatile=True)
    print("3 : " )
    print(inputs)
    '''
    inputs = inputs.cuda()
    outputs = net(inputs)
    outputs_avg = outputs.view(1, -1).mean(0)  # avg over crops
    
    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)
    print("score :", score)
    print("predicted :",predicted)
    '''
    plt.rcParams['figure.figsize'] = (13.5,5.5)
    axes=plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.xlabel('Input Image', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()


    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

    plt.subplot(1, 3, 2)
    ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
    width = 0.4       # the width of the bars: can also be len(x) sequence
    color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
    for i in range(len(class_names)):
        plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
    plt.title("Classification results ",fontsize=20)
    plt.xlabel(" Expression Category ",fontsize=16)
    plt.ylabel(" Classification Score ",fontsize=16)
    plt.xticks(ind, class_names, rotation=45, fontsize=14)

    axes=plt.subplot(1, 3, 3)
    emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
    plt.imshow(emojis_img)
    plt.xlabel('Emoji Expression', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()
    # show emojis
    
    #plt.show()
    plt.savefig(os.path.join('images/results/'+str(a)+'.png'))
    plt.close()
    '''
    
    print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))
    if str(class_names[int(predicted.cpu().numpy())]) == 'disgusted' :
        a+=1
    total +=1
    
print("correct : " + str(a))
print("acc :", a/total*100)



