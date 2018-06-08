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

import AF_1
import AF_2
import AF_3
import Incep
import Hydraplus
import dataload

from torch.autograd import Variable

from visdom import Visdom
import numpy as np
viz = Visdom()
win = viz.line(
    Y=np.array([0.2]),
    name="1"
)










import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m',help = "choose model",choices = ['MNet','AF1','AF2','AF3','HP'])
parser.add_argument('-p',help = "load weight path",default = None)
parser.add_argument('-mpath',help = "load MNet weight path",default = None)
parser.add_argument('-af1path',help = "load AF1 weight path",default = None)
parser.add_argument('-af2path',help = "load AF2 weight path",default = None)
parser.add_argument('-af3path',help = "load AF3 weight path",default = None)
parser.add_argument()
args = parser.parse_args()


def checkpoint(epoch):
    if not os.path.exists(args.m):
        os.mkdir(arg.m)
    savePath = arg.m+"/checkpoint_epoch_{}".format(epoch)
    torch.save(net.state_dict(),savePath)

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)




mytransform = transforms.Compose([
    
    transforms.RandomHorizontalFlip(),
    transforms.Resize((299,299)),       #TODO:maybe need to change1
    transforms.ToTensor(),            
    ]
)

# torch.utils.data.DataLoader
set = dataload.myImageFloder(root = "./data/PA-100K/release_data/release_data", 
                    label = "./data/PA-100K/annotation/annotation.mat", 
                    transform = mytransform,
                    mode = 'train' )
imgLoader = torch.utils.data.DataLoader(
         set, 
         batch_size= 16, shuffle= True, num_workers= 2)


print("image numbers %d"%len(set) )



if args.m == 'AF1':
    net = AF_1.AF1()

    if not args.p:
        path = args.p
        net.load_state_dict(torch.load(path))
    else:
        #MPath = "./checkpoint1/checkpoint_epoch_60"                 #FIXME:
        net.MNet.load_state_dict(torch.load(args.mpath))

    for param in net.MNet.parameters():
        param.requires_grad = False
if args.m == 'AF2':
    net = AF_2.AF2()

    if not args.p:
        path = args.p
        net.load_state_dict(torch.load(path))
    else:
        #MPath = "./checkpoint1/checkpoint_epoch_60"                 #FIXME:
        net.MNet.load_state_dict(torch.load(args.mpath))

    for param in net.MNet.parameters():
        param.requires_grad = False   
if args.m == 'AF3':
    net = AF_3.AF3()


    if not args.p:
        path = args.p
        net.load_state_dict(torch.load(path))
    else:
        #MPath = "./checkpoint1/checkpoint_epoch_60"                 #FIXME:
        net.MNet.load_state_dict(torch.load(args.mpath))

    for param in net.MNet.parameters():
        param.requires_grad = False
if args.m == 'HP':
    net = Hydraplus.HP()
    if not args.p:
        path = args.p
        net.load_state_dict(torch.load(path))
    else:
        net.MNet.load_state_dict(torch.load(args.mpath))
        net.AF1.load_state_dict(torch.load(args.af1path))
        net.AF2.load_state_dict(torch.load(args.af2path))
        net.AF3.load_state_dict(torch.load(args.af3path))


    for param in net.MNet.parameters():
    param.requires_grad = False

    for param in net.AF1.parameters():
        param.requires_grad = False

    for param in net.AF2.parameters():
        param.requires_grad = False

    for param in net.AF3.parameters():
        param.requires_grad = False
if args.m == 'MNet':
    net = Incep.Inception3()
    if not args.p:
        path = args.p
        net.load_state_dict(torch.load(path))
    else:
        net.apply(weight_init)


net.train()
net.cuda()




#print(net.parameters())
#print(net)


#[u'Female', u'AgeOver60', u'Age18-60', u'AgeLess18', u'Front', u'Side', u'Back', u'Hat', 
# u'Glasses', u'HandBag', u'ShoulderBag', u'Backpack', u'HoldObjectsInFront', u'ShortSleeve', u'LongSleeve', u'UpperStride',
#  u'UpperLogo', u'UpperPlaid', u'UpperSplice', u'LowerStripe', u'LowerPattern', u'LongCoat', u'Trousers', u'Shorts', 
# u'Skirt&Dress', u'boots']


weight = torch.Tensor([1.7226262226969686, 2.6802565029531618, 1.0682133644154836, 2.580801475214588, 
1.8984257687918218, 2.046590013290684, 1.9017984669155032, 2.6014006200502586, 
2.272458988404639, 2.2625669787021203, 2.245380512162444, 2.3452980639899033, 
2.692210221689372, 1.5128949487853383, 1.7967419553099035, 2.5832221110933764, 
2.3302195718894034, 2.438480257574324, 2.6012705532709526, 2.704589108443237, 
2.6704246374231753, 2.6426970354162505, 1.3377813061118478, 2.284449325734624, 
2.417810793601295, 2.7015143874115033])


criterion = nn.BCEWithLogitsLoss(weight = weight)          #TODO:1.learn 2. weight
criterion.cuda()

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,net.parameters()), lr=0.001,momentum=0.9)

running_loss = 0.0
for epoch in range(1000):
    for i, data in enumerate(imgLoader, 0):
            # get the inputs
            inputs, labels = data
            
            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            


            # forward + backward + optimize
            outputs = net(inputs)
            
            #print(outputs)

            loss = criterion(outputs, labels)
            #print(loss)
            loss.backward()        
            optimizer.step()
            
            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 0: # print every 1000 mini-batches
                print('[ %d %5d] loss: %.6f' % ( epoch,i+1, running_loss / 100))
                viz.updateTrace(
                    X=np.array([epoch+i/8000.0]),
                    Y=np.array([running_loss]),
                    win=win,
                    name="1"
                )
                running_loss = 0.0

    if epoch % 5 == 0:
        checkpoint(epoch)
        '''for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.95'''


    