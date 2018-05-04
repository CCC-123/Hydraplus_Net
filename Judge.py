import os
import torch
import torch.utils.data as data
from PIL import Image

import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt
import torch.utils
import torchvision
import torchvision.models as models


import scipy.io as scio
import torchvision.transforms as transforms

import sys

import Hydraplus

class_names = ['Female','AgeOver60','Age18-60','AgeLess18','Front',
                'Side','Back','Hat','Glasses','HandBag',
                'ShoulderBag','Backpack','HoldObjectsInFront','ShortSleeve','LongSleeve',
                'UpperStride','UpperLogo','UpperPlaid','UpperSplice','LowerStripe',
                'LowerPattern','LongCoat','Trousers','Shorts','Skirt&Dress',
                'boots']
class_len = 35





def Judge(imgpath) :
        mytransform = transforms.Compose([
            transforms.Resize((299,299)),
            transforms.ToTensor(),            # mmb
            ]
        )

        path = "/home/ubuntu/Desktop/Hydraplus/checkpoint5/checkpoint_epoch_40" 
        net = Hydraplus.HP()
        net.load_state_dict(torch.load(path))
        net.eval()
        net.cuda()

        images = Image.open(imgpath).convert('RGB')
        images = mytransform(images)
        images = images.view(1,3,299,299)
        inputs = Variable(images,volatile = True).cuda()
        outputs = net(inputs)
            
        count = 0
        ret = []
        for item in outputs[0]:
            if item.data[0] > 0:
                ret.append(class_names[count])
            count = count + 1
        return ret
















