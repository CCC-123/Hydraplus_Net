import os
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import torch.utils
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

import math

import dataload

mytransform = transforms.Compose([

    transforms.ToTensor(),            
    ]
)

set = dataload.myImageFloder(root = "./data/PA-100K/release_data/release_data", 
                            label = "./data/PA-100K/annotation/annotation.mat", 
                            transform = mytransform,
                            mode = 'train' )

imgLoader = torch.utils.data.DataLoader(
         set, 
         batch_size= 1, shuffle= True, num_workers= 2)


print("image numbers %s" %len(set)) 


count = 0

dataiter = iter(imgLoader)
P  = [0.0] * 26
N  = [0.0] * 26
while count < 80000:
    images,labels = dataiter.next()
    for i in range(26):         
            if labels[0][i] == 1:
                P[i] = P[i] + 1
            else :

                N[i] = N[i] + 1   
    count = count + 1



print(P)
print(N)

w = []
for i in range(26):
    w.append(math.pow(math.e,N[i]/80000.0))

print(w)

f = open("loss weight",'w')
for weight in w:
    f.write("%f "%weight)

f.close()