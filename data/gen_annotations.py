# -*- coding: utf-8 -*-
# In[]
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# In[]
x_tr, y_tr = torch.load('./raw/mnist_train.pt')
x_te, y_te = torch.load('./raw/mnist_test.pt')

# In[]
def find_annot_coord(img=None, ):
    import torch
    if type(img)!= torch.Tensor:
        raise Exception("Unsupported type of img: ",type(img))
    top_row=0
    bottom_row=img.shape[0]-1
    left_col=0
    right_col=img.shape[1]-1
    for i in range(top_row,bottom_row):
        non_zero_in_row=img[i,:].nonzero()# if type(img) is np.ndarry,then add 'non_zero_in_row=non_zero_in_row[0]'
        if non_zero_in_row.shape[0]>0:
            top_row=max(0,i-1)
            break
    for i in range(bottom_row,top_row,-1):
        non_zero_in_row=img[i,:].nonzero()
        if non_zero_in_row.shape[0]>0:
            bottom_row=min(img.shape[0]-1,i+1)
            break
    for i in range(left_col,right_col):
        non_zero_in_col=img[top_row:bottom_row,i].nonzero()
        if non_zero_in_col.shape[0]>0:
            left_col=max(0,i-1)
            break
    for i in range(right_col,left_col,-1):
        non_zero_in_col=img[top_row:bottom_row,i].nonzero()
        if non_zero_in_col.shape[0]>0:
            right_col=min(img.shape[1]-1,i+1)
            break
    return top_row,bottom_row,left_col,right_col

# In[]
def find_annot_coords_for_multi_samples(x,y):
    annot_coords=torch.zeros([y.shape[0],4,2])
    for i in range(0,y.shape[0]):
        top_row,bottom_row,left_col,right_col=find_annot_coord(x[i])
        annot_coords[i,:,:]=torch.tensor([
            [top_row,left_col],
            [top_row,right_col],
            [bottom_row,left_col],
            [bottom_row,right_col]])
    return annot_coords
annot_coords_tr=find_annot_coords_for_multi_samples(x_tr,y_tr)
annot_coords_te=find_annot_coords_for_multi_samples(x_te,y_te)
# In[]
torch.save(annot_coords_tr, './raw/annot_coords_tr.pt')
torch.save(annot_coords_te, './raw/annot_coords_te.pt')
# In[]


