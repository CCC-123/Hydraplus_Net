import os
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import torch.utils
import torchvision
import torch.nn as nn

from torch.autograd import Variable

import scipy.io as scio
import torchvision.transforms as transforms

import argparse

import dataload
import Hydraplus
import Incep
import AF_1
import AF_2
import AF_3

import time

parser = argparse.ArgumentParser()
parser.add_argument('-m',help = "choose model",choices = ['AF1','AF2','AF3','HP','MNet'],required = True)
parser.add_argument('-p',help = 'wight file path',required = True)
args = parser.parse_args()







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


print len(set)

mat = scio.loadmat("./data/PA-100K/annotation/annotation.mat")
att = mat["attributes"]

count = 0
classes = []
for c in att:
    classes.append(c[0][0])
    count = count + 1

path = args.p
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

count = 0

TP = [0.0] * 26
P  = [0.0] * 26
TN = [0.0] * 26
N  = [0.0] * 26

Acc = 0.0
Prec = 0.0
Rec = 0.0
while count < 10000:
    images,labels = dataiter.next()
    inputs, labels = Variable(images,volatile = True).cuda(), Variable(labels).cuda()
    #a = time.time()
    outputs = net(inputs)
    #b = time.time()
    #print(b-a)    
    Yandf = 0.1
    Yorf = 0.1
    Y = 0.1
    f = 0.1

    i = 0
    for item in outputs[0]:
            if item.data[0] > 0 :
                f = f + 1
                Yorf = Yorf + 1
                if labels[0][i].data[0] == 1:
                    TP[i] = TP[i] + 1
                    P[i] = P[i] + 1
                    Y = Y + 1
                    Yandf = Yandf + 1
                else : 
                    N[i] = N[i]  + 1
            else :
                if labels[0][i].data[0] == 0 :
                    TN[i] = TN[i] + 1
                    N[i] = N[i] + 1
                else:
                    P[i] = P[i] + 1
                    Yorf = Yorf + 1
                    Y = Y + 1
            i = i + 1 
    Acc = Acc +Yandf/Yorf
    Prec = Prec + Yandf/f
    Rec = Rec + Yandf/Y
    if count % 1000 == 0:
        print(count)      
    count = count + 1

Accuracy = 0
print(TP)
print(TN)
print(P)
print(N)
for l in range(26):
    print( "%s : %f" %(classes[l],(TP[l]/P[l] + TN[l]/N[l])/2))
    Accuracy =  TP[l]/P[l] + TN[l]/N[l] + Accuracy
meanAccuracy = Accuracy / 52

print("path: %s mA: %f"%(path,meanAccuracy))

Acc = Acc/10000
Prec = Prec/10000
Rec = Rec/10000
F1 = 2 * Prec * Rec / (Prec + Rec)

print("ACC: %f"%(Acc))
print("Prec: %f"%(Prec))
print("Rec: %f"%(Rec))
print("F1: %f"%(F1))
