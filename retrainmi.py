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

import inception_v3
from torch.autograd import Variable


def default_loader(path):
    return Image.open(path).convert('RGB')

class myImageFloder(data.Dataset):
    def __init__(self, root, label, transform = None, target_transform=None, loader=default_loader):

        fn = scio.loadmat(label)
        imgs = []
        testlabel = fn['train_label']
        testimg = fn['train_images_name']
        count = 0
        for name in testimg:
            #print name[0][0]
            if os.path.isfile(os.path.join(root,name[0][0])):
                #imgs.append((name[0][0],[x*2-1 for x in testlabel[count]]))   (-1,1)
                imgs.append((name[0][0],testlabel[count]))
                if testlabel[count][21] == 1:
                    for i in range(20):
                        imgs.append((name[0][0],testlabel[count]))
                if testlabel[count][12] == 1:
                    for i in range(90):
                        imgs.append((name[0][0],testlabel[count]))
                if testlabel[count][25] == 1:
                    for i in range(150):
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
    if not os.path.exists("./checkpoint/retrainmi"):
        os.mkdir("./checkpoint/retrainmi")
    path = "./checkpoint/retrainmi/checkpoint_epoch_{}".format(epoch)
    torch.save(net.state_dict(),path)







mytransform = transforms.Compose([
    
    transforms.Resize((299,299)),       #TODO:maybe need to change1
    transforms.ToTensor(),            # mmb,
    ]
)

# torch.utils.data.DataLoader
set = myImageFloder(root = "./data/PA-100K/release_data/release_data", label = "./data/PA-100K/annotation/annotation.mat", transform = mytransform )
imgLoader = torch.utils.data.DataLoader(
         set, 
         batch_size= 10, shuffle = True, num_workers = 2)


print len(set)


'''dataiter = iter(imgLoader)
images,labels = dataiter.next()
imshow(images)'''

historyPath = "./history/checkpoint/3.27-0.643"                     #FIXME:PATH
net = inception_v3.Inception3()
net.load_state_dict(torch.load(historyPath))
net.cuda()

#print(net)


#[u'Female', u'AgeOver60', u'Age18-60', u'AgeLess18', u'Front', u'Side', u'Back', u'Hat', 
# u'Glasses', u'HandBag', u'ShoulderBag', u'Backpack', u'HoldObjectsInFront', u'ShortSleeve', u'LongSleeve', u'UpperStride',
#  u'UpperLogo', u'UpperPlaid', u'UpperSplice', u'LowerStripe', u'LowerPattern', u'LongCoat', u'Trousers', u'Shorts', 
# u'Skirt&Dress', u'boots']


weight = torch.FloatTensor(1,26)

weight[0][0] = 1.84
weight[0][1] = 2.64 
weight[0][2] = 1.03
weight[0][3] = 2.69 
weight[0][4] = 2.01
weight[0][5] = 1.82
weight[0][6] = 2.01
weight[0][7] = 2.64 
weight[0][8] = 2.12
weight[0][9] = 2.34 
weight[0][10] = 2.23
weight[0][11] = 2.34
weight[0][12] = 2.69 
weight[0][13] = 1.75
weight[0][14] = 1.55
weight[0][15] = 2.56
weight[0][16] = 2.41
weight[0][17] = 2.41
weight[0][18] = 2.61
weight[0][19] = 2.71 
weight[0][20] = 2.69 
weight[0][21] = 2.61 
weight[0][22] = 1.23
weight[0][23] = 2.39
weight[0][24] = 2.50 
weight[0][25] = 2.69   

criterion = nn.BCEWithLogitsLoss(weight = weight)          #TODO:1.learn 2. weight
criterion.cuda()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
            if i % 1000 == 999: # print every 2000 mini-batches
                print('[ %d %5d] loss: %.6f' % ( epoch,i+1, running_loss / 1000))
                running_loss = 0.0
    
    checkpoint(epoch)


    