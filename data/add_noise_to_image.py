# -*- coding: utf-8 -*-

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# In[]
x_tr, y_tr = torch.load('./raw/mnist_train.pt')
x_te, y_te = torch.load('./raw/mnist_test.pt')
# In[]

img=x_tr[0]
plt.imshow(np.uint8(img), cmap='gray')
# In[]
noise=torch.normal(mean=0.2, std=0.1, size=(28,28),out=None)*255.0
# In[]
new_img=(img+noise).clamp(min=0, max=255)
new_img=new_img.type(torch.uint8)
plt.imshow(np.uint8(new_img), cmap='gray')

# In[]
def add_noise_to_dataset(x,y):
    for i in range(0,y.shape[0]):
        noise=torch.normal(mean=0.2, std=0.1, size=(28,28),out=None)*255.0
        x[i]=(x[i]+noise).clamp(min=0, max=255)
        x[i]=x[i].type(torch.uint8)
add_noise_to_dataset(x_tr, y_tr)
add_noise_to_dataset(x_te, y_te)

# In[]
torch.save((x_tr, y_tr), './raw/noisy_mnist_train.pt')
torch.save((x_te, y_te), './raw/noisy_mnist_test.pt')
# In[]
noisy_x_tr, noisy_y_tr = torch.load('./noisy_mnist_train.pt')
noisy_x_te, noisy_y_te = torch.load('./noisy_mnist_test.pt')

# In[]
img=x_te[0]
plt.imshow(np.uint8(img), cmap='gray')