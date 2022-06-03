# -*- coding: utf-8 -*-

import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
# In[]
x_tr, y_tr = torch.load('./raw/mnist_train.pt')

# In[]
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
img=np.uint8(x_tr[0])
img_show(img)
# In[]
#cv2.imwrite('5.jpg', img)

#pil_img = Image.merge('L', img)
plt.imshow(img, cmap='gray')
plt.savefig('5.png')
# In
#plt.show()
