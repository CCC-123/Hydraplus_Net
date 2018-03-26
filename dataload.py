import os
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import torch.utils
import torchvision

import scipy.io as scio
import torchvision.transforms as transforms




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

mytransform = transforms.Compose([
    
    transforms.CenterCrop(224),
    transforms.ToTensor(),            # mmb
    ]
)

# torch.utils.data.DataLoader
set = myImageFloder(root = "./data/PA-100K/release_data/release_data", label = "./data/PA-100K/annotation/annotation.mat", transform = mytransform )
imgLoader = torch.utils.data.DataLoader(
         set, 
         batch_size= 2, shuffle= False, num_workers= 2)


print len(set)


dataiter = iter(imgLoader)
images,labels = dataiter.next()
imshow(images)