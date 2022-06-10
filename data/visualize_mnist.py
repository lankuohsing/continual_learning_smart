# -*- coding: utf-8 -*-

import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import random
# In[]
"""
x_tr.shape: [60000,28,28], torch.Tensor
y_tr.shape: 60000, torch.Tensor
"""
x_tr, y_tr = torch.load('./raw/mnist_train.pt')
# In[]
"""
each ele stores 4 coordinates of the annotation rectangle.
[left_top, right_top, left_bottom,right_bottom]
"""
annot_coords_tr=np.zeros((y_tr.shape[0],4,2))
# In[]
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
img=np.uint8(copy.deepcopy(x_tr[1]))
#img_show(img)
# In[]
#cv2.imwrite('5.jpg', img)

#pil_img = Image.merge('L', img)
plt.imshow(img, cmap='gray')
plt.savefig('0.png')
# In
#plt.show()

# In[]
top_row=0
bottom_row=img.shape[0]-1
left_col=0
right_col=img.shape[1]-1
for i in range(top_row,bottom_row):
    non_zero_in_row=img[i,:].nonzero()[0]
    if non_zero_in_row.shape[0]>0:
        top_row=max(0,i-1)
        break
for i in range(bottom_row,top_row,-1):
    non_zero_in_row=img[i,:].nonzero()[0]
    if non_zero_in_row.shape[0]>0:
        bottom_row=min(img.shape[0]-1,i+1)
        break
for i in range(left_col,right_col):
    non_zero_in_col=img[top_row:bottom_row,i].nonzero()[0]
    if non_zero_in_col.shape[0]>0:
        left_col=max(0,i-1)
        break
for i in range(right_col,left_col,-1):
    non_zero_in_col=img[top_row:bottom_row,i].nonzero()[0]
    if non_zero_in_col.shape[0]>0:
        right_col=min(img.shape[1]-1,i+1)
        break
# In]
img[top_row,left_col:right_col+1]=255.0
img[bottom_row,left_col:right_col+1]=255.0
img[top_row:bottom_row+1,left_col]=255.0
img[top_row:bottom_row+1,right_col]=255.0
plt.imshow(img, cmap='gray')
# In[]
coord_i=np.array([
        [top_row,left_col],
        [top_row,right_col],
        [bottom_row,left_col],
        [bottom_row,right_col]])
# In[]
"""

"""
img[coord_i[0][0],coord_i[0][1]:coord_i[1][1]+1]=255.0
img[coord_i[2][0],coord_i[2][1]:coord_i[3][1]+1]=255.0
img[coord_i[0][0]:coord_i[2][0]+1,coord_i[0][1]]=255.0
img[coord_i[0][0]:coord_i[2][0]+1,coord_i[1][1]]=255.0
plt.imshow(img, cmap='gray')
# In[]
noise=np.random.normal(loc=0.2, scale=0.1, size=img.shape)*255.0
new_img=np.uint8(np.clip(img+noise,0,255.0))
plt.imshow(new_img, cmap='gray')
plt.savefig('0.png')
# In[]
# 一维矩阵
x= np.arange(12)
print(np.clip(x,3,8))
# 多维矩阵
y= np.arange(12).reshape(3,4)
print(np.clip(y,3,8))
