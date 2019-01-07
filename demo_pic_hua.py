
# coding: utf-8
# CMU openpose的pytorch 版本，只有推理，没有训练。by hua
import os
import cv2
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.autograd import Variable
from scipy.ndimage.filters import gaussian_filter
# openpose的 功能接口函数
from utils import *
from pose_estimation import *

import import ipdb

# ### 加载测试图片
# In[28]:


use_gpu = True

test_image = './COCO_val2014_000000000474.jpg'
img_ori = cv2.imread(test_image) # B,G,R order

# display the validation pics
plt.figure(figsize=(12, 8))
plt.imshow(img_ori[...,::-1])


# ### 加载模型
state_dict = torch.load('./models/coco_pose_iter_440000.pth.tar')['state_dict']

model_pose = get_pose_model()
model_pose.load_state_dict(state_dict)
model_pose.float()
model_pose.eval()


# ### 使用 GPU

if use_gpu:
    model_pose.cuda()
    model_pose = torch.nn.DataParallel(model_pose, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# ### 获取 PAF 和 Heat Map

scale_param = [0.5, 1.0, 1.5, 2.0]
paf_info, heatmap_info = get_paf_and_heatmap(model_pose, img_ori, scale_param)


# ### 提取 Heat Map 的关键点

peaks = extract_heatmap_info(heatmap_info)


# ### 提取 PAF 信息
sp_k, con_all = extract_paf_info(img_ori, paf_info, peaks)


# In[34]:


subsets, candidates = get_subsets(con_all, sp_k, peaks)


# ### 标识关键点

# In[35]:


get_ipython().run_cell_magic('time', '', "\nsubsets, img_points = draw_key_point(subsets, peaks, img_ori)\nimg_canvas = link_key_point(img_points, candidates, subsets)\n\ncv2.imwrite('result.png', img_canvas)   \n\nplt.figure(figsize=(12, 8))\n\nplt.subplot(1, 2, 1)\nplt.imshow(img_points[...,::-1])\n\nplt.subplot(1, 2, 2)\nplt.imshow(img_canvas[...,::-1])")

