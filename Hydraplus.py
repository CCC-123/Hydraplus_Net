import torch
import torch.nn as nn
import torch.nn.functional as F
import Incep
import AF_1
import AF_2
import AF_3

class HP(nn.Module):

    def __init__(self,num_classes = 26):
        super(HP,self).__init__()
        self.MNet = Incep.Inception3(nfc = True)
        self.AF1 = AF_1.AF1(ret = True)
        self.AF2 = AF_2.AF2(ret = True)
        self.AF3 = AF_3.AF3(ret = True)

        self.fc = nn.Linear(2048*73,num_classes)

    def forward(self,x):
        F0 = self.MNet(x)
        F1 = self.AF1(x)
        F2 = self.AF2(x)
        F3 = self.AF3(x)

        ret = torch.cat((F0,F1,F2,F3),dim = 1)
        # 8 x 8 x (2048x(24x3 + 1))

        ret = F.avg_pool2d(ret,kernel_size = 8)

        # 1 x 1 x (2048 x 73)

        ret = F.dropout(ret, training=self.training)
        # 1 x 1 x (2048 x 73)
        ret = ret.view(ret.size(0), -1)
        # 2048 x 73

        ret = self.fc(ret)
        # (num_classes)



        return ret