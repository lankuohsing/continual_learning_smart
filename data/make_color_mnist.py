# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.datasets as datasets
import sys
import numpy as np
import torch.utils.data as utils
from tqdm import tqdm
from colour import Color
import os
from os.path import join as oj
import matplotlib.pyplot as plt
# In[]
np.random.seed(0)
red = Color("red")
colors = list(red.range_to(Color("purple"),10))
colors = [np.asarray(x.get_rgb()) for x in colors]

# In[]
mnist_trainset = datasets.MNIST(root='../../data', train=True, download=False, transform=None)
num_train = int(len(mnist_trainset)*.9)
num_val = len(mnist_trainset)  - num_train
torch.manual_seed(0);
train_dataset, val_dataset,= torch.utils.data.random_split(mnist_trainset, [num_train, num_val])
# In[]
num_samples = len(train_dataset)
color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)
color_y = np.empty(num_samples, dtype = np.int16)
for i in tqdm(range(num_samples)):
    my_color  = colors[train_dataset.dataset.train_labels[train_dataset.indices[i]].item()]
    color_x[i ] = train_dataset.dataset.data[train_dataset.indices[i]].numpy().astype(np.float32)[np.newaxis]*my_color[:, None, None]
    color_y[i] = train_dataset.dataset.targets[train_dataset.indices[i]]

# In[]
def show_image_3d(pic_0):
    # (3,32,32)
    if len(pic_0.shape)==3:
#        image = pic_0.transpose(1,2,0)# (32,32,3)
        image=np.rollaxis(pic_0, 0, 3)# (32,32,3)
#        image1=np.rollaxis(image, 2, 0)# (3,32,32)
#        image=np.rollaxis(image1, 0, 3)# (32,32,3)
    else:
        if len(pic_0.shape)==2:
            image=pic_0
        else:
            print("wrong shape")
            raise
#    print(image.shape)
    plt.imshow(image)
    plt.show()
img=color_x[0]
show_image_3d(img)
# In[]
up_row=0
down_row=img.shape[1]-1
left_col=0
right_col=img.shape[2]-1
for i in range(up_row,down_row):
    non_zero_in_row=img[0][i,:].nonzero()
    if non_zero_in_row[0].shape[0]>0:
        up_row=max(0,i-1)
        break
for i in range(down_row,up_row,-1):
    non_zero_in_row=img[0][i,:].nonzero()
    if non_zero_in_row[0].shape[0]>0:
        down_row=min(img.shape[1]-1,i+1)
        break
for i in range(left_col,right_col):
    non_zero_in_col=img[0][:,i].nonzero()
    if non_zero_in_col[0].shape[0]>0:
        left_col=max(0,i-1)
        break
for i in range(right_col,left_col,-1):
    non_zero_in_col=img[0][:,i].nonzero()
    if non_zero_in_col[0].shape[0]>0:
        right_col=min(img.shape[2]-1,i+1)
        break
# In]
for i in range(0,3):
    img[i][up_row,left_col:right_col+1]=255.0
    img[i][down_row,left_col:right_col+1]=255.0
    img[i][up_row:down_row+1,left_col]=255.0
    img[i][up_row:down_row+1,right_col]=255.0
show_image_3d(img)
# In[]
show_image_3d(img)
# In[]
os.makedirs("../../data/ColorMNIST", exist_ok = True)
np.save(oj("../../data/ColorMNIST", "train_x.npy"), color_x)
np.save(oj("../../data/ColorMNIST", "train_y.npy"), color_y)


num_samples = len(val_dataset)
color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)
color_y = np.empty(num_samples, dtype = np.int16)
for i in tqdm(range(num_samples)):
    my_color  = colors[9-val_dataset.dataset.train_labels[val_dataset.indices[i]].item()]
    color_x[i ] = val_dataset.dataset.data[val_dataset.indices[i]].numpy().astype(np.float32)[np.newaxis]*my_color[:, None, None]
    color_y[i] = val_dataset.dataset.targets[val_dataset.indices[i]]
os.makedirs("../../data/ColorMNIST", exist_ok = True)
np.save(oj("../../data/ColorMNIST", "val_x.npy"), color_x)
np.save(oj("../../data/ColorMNIST", "val_y.npy"), color_y)




mnist_trainset = datasets.MNIST(root='../../data', train=False, download=True, transform=None)
num_samples = len(mnist_trainset)
color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)

color_y = mnist_trainset.train_labels.numpy().copy()
for i in tqdm(range(num_samples)):
    color_x[i ] = mnist_trainset.data[i].numpy().astype(np.float32)[np.newaxis]*colors[9-color_y[i]] [:, None, None]

np.save(oj("../../data/ColorMNIST", "test_x.npy"),  color_x)
np.save(oj("../../data/ColorMNIST", "test_y.npy"), color_y)