import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras
from sklearn import preprocessing
import numpy as np
from sklearn.manifold import TSNE
import sklearn
import argparse

parser = argparse.ArgumentParser(description='TSNE')
parser.add_argument('--per', type=int, default=10, help='perlexity')
parser.add_argument('--bs', type=int, default=1000, help='batch size')
parser.add_argument('--txtname', type=str,  help='batch size')
opt = parser.parse_args()
data_generator = ImageDataGenerator(rescale=1./255)

train_generator = data_generator.flow_from_directory(
    './Datasets/FER2013/train/',
    target_size=(48,48),
    batch_size=opt.bs,
    class_mode='binary',
    color_mode='grayscale',
    shuffle = True)
x_train,y_train = train_generator.next()
print(x_train.shape[0] ,x_train.shape[1] , x_train.shape[2])
dim_x = 48*48
dim_y = 128
print(x_train.shape)
print(y_train.shape)
x_train = x_train.reshape(x_train.shape[0],dim_x).astype(np.float32)

#scaler = preprocessing.MinMaxScaler().fit(x_train)
#x_train = scaler.transform(x_train)
#scaler = preprocessing.MinMaxScaler().fit(y_train)
#y_train = scaler.transform(y_train)
print(x_train.shape)
print(y_train.shape)

print(x_train.shape)
print(y_train.shape)

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

def draw_scatter(x, n_class, colors):
    sns.palplot(sns.color_palette("hls", n_class))
    palette = np.array(sns.color_palette("hls", n_class))
    
    f = plt.figure(figsize=(14, 14))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    #plt.show()
    plt.savefig("./"+ opt.txtname+".png",dpi=300)
    
tsne_train_xs = TSNE(learning_rate=200,metric='euclidean',init='random',random_state=0,perplexity=opt.per
                    ,n_iter = 1000).fit_transform(x_train)
print("\ncomplete")
xs = tsne_train_xs[:,0]
xy = tsne_train_xs[:,1]

draw_scatter(tsne_train_xs, 7 , y_train )