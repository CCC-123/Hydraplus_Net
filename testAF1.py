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
from torch.autograd import Variable


def default_loader(path):
    return Image.open(path).convert('RGB')

class myImageFloder(data.Dataset):
    def __init__(self, root, label, transform = None, target_transform=None, loader=default_loader):

        fn = scio.loadmat(label)
        imgs = []
        testlabel = fn['test_label']
        testimg = fn['test_images_name']
        count = 0
        for name in testimg:
            #print name[0][0]
            if os.path.isfile(os.path.join(root,name[0][0])):
                #imgs.append((name[0][0],[x*2-1 for x in testlabel[count]]))   (-1,1)
                imgs.append((name[0][0],testlabel[count]))
            count=count+1

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.classes = fn['attributes']

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.imgs)
    
    def getName(self):
        return self.classes

def imshow(imgs):
    grid = torchvision.utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose(1,2,0))
    plt.title("bat")
    plt.show()


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    path = "checkpoint_epoch_{}".format(epoch)
    torch.save(net,path)



mytransform = transforms.Compose([
    
    transforms.Resize((299,299)),       #FIXME:resize
    transforms.ToTensor(),            # mmb,
    ]
)

# torch.utils.data.DataLoader
set = myImageFloder(root = "./data/PA-100K/release_data/release_data", label = "./data/PA-100K/annotation/annotation.mat", transform = mytransform )
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
print(classes)

path = "./checkpoint2/checkpoint_epoch_15"                     #FIXME:PATH
net = AF_1.AF1()
net.load_state_dict(torch.load(path))
net.eval()
net.cuda()

dataiter = iter(imgLoader)

count = 0

TP = [0.0] * 26
P  = [0.0] * 26
TN = [0.0] * 26
N  = [0.0] * 26
while count < 10000:
    images,labels = dataiter.next()
    inputs, labels = Variable(images,volatile = True).cuda(), Variable(labels).cuda()
    outputs = net(inputs)
                


    i = 0
    for item in outputs[0]:
            if item.data[0] > 0 :
                if labels[0][i].data[0] == 1:
                    TP[i] = TP[i] + 1
                    P[i] = P[i] + 1
                else : 
                    N[i] = N[i]  + 1
            else :
                if labels[0][i].data[0] == 0 :
                    TN[i] = TN[i] + 1
                    N[i] = N[i] + 1
                else:
                    P[i] = P[i] + 1
            i = i + 1 
    if count % 1000 == 0:
        print(count)      
    count = count + 1

Accuracy = 0
print(TP)
print(TN)
print(P)
print(N)
for l in range(26):
    print(l, ":" , (TP[l]/P[l] + TN[l]/N[l])/2)
    Accuracy =  TP[l]/P[l] + TN[l]/N[l] + Accuracy
meanAccuracy = Accuracy / 52

print("mA: %f",meanAccuracy)


