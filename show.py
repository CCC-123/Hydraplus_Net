import os
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import torch.utils
import torchvision
import torch.nn as nn

import scipy.io as scio
import torchvision.transforms as transforms
from torch.autograd import Variable

import argparse

import dataload
import Hydraplus
import Incep
import AF_1
import AF_2
import AF_3

parser = argparse.ArgumentParser()
parser.add_argument('-m',help = "choose model",choices = ['AF1','AF2','AF3','HP','MNet'],required = True)
parser.add_argument('-p',help = 'wight file path',required = True)
args = parser.parse_args()



def imshow(imgs):
    grid = torchvision.utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose(1,2,0))
    plt.title("show")
    plt.show()





mytransform = transforms.Compose([
    
    transforms.Resize((299,299)),       #FIXME:resize
    transforms.ToTensor(),            # mmb,
    ]
)

# torch.utils.data.DataLoader
set = dataload.myImageFloder(root = "./data/PA-100K/release_data/release_data", 
                            label = "./data/PA-100K/annotation/annotation.mat", 
                            transform = mytransform,
                            mode = 'test' )
imgLoader = torch.utils.data.DataLoader(
         set, 
         batch_size= 1, shuffle= True, num_workers= 2)


print("image numbers: %d"%len(set)) 


mat = scio.loadmat("./data/PA-100K/annotation/annotation.mat")
att = mat["attributes"]

count = 0
classes = []
for c in att:
    classes.append(c[0][0])
    count = count + 1


path = args.p                     #FIXME:PATH
if args.m == 'AF1':
    net = AF_1.AF1()
if args.m == 'AF2':
    net = AF_2.AF2()
if args.m == 'AF3':
    net = AF_3.AF3()
if args.m == 'HP':
    net = Hydraplus.HP()
if args.m == 'MNet':
    net = Incep.Inception3()


net.load_state_dict(torch.load(path))
net.eval()
net.cuda()

dataiter = iter(imgLoader)
x = " "
while True and x != 'n':
    images,labels = dataiter.next()


    inputs, labels = Variable(images).cuda(), Variable(labels).cuda()


    outputs = net(inputs)
                
    print(outputs)
    print(labels)

    count = 0

    for item in outputs[0]:

            if item.data[0] > 0:
                print(classes[count])
            count = count + 1
    print('\n')
    print("Enter to continue,input 'n' to break")
    count = 0
    for item in labels[0]:
        if item.data[0] > 0:
            print(classes[count])
        count = count + 1



    imshow(images)   
    x = raw_input()


