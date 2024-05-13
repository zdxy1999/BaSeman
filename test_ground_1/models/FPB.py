import torch
import torch.nn as nn

##123455678

class ATP_Block(nn.Module):
    def __init__(self, group_size, in_channels, out_channels,dropout=0.5):
        super(ATP_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),stride=1,groups=group_size)
        self.conv2 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),stride=1,groups=group_size,padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),stride=1,dilation=2,padding=2)
        #self.conv4 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),stride=1,dilation=3,padding=3)
        #self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1,dilation=4, padding=4)
        self.weight = nn.parameter.Parameter(torch.Tensor(3), requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(stride=2,kernel_size=(2,2))

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        # x4 = self.conv4(x)
        # x5 = self.conv5(x)

        weight = torch.softmax(self.weight, dim=0)
        x = x1*weight[0]+x2*weight[1]+x3*weight[2]#+x4*weight[3]+x5*weight[4]
        self.dropout(x)
        x = self.act(x)
        x = self.pool(x)

        return x

class AttMapGeneratorP(nn.Module):
    def __init__(self, in_channel,class_num,dims=[64,128],pool_size=7,dropout=0.5):
        super(AttMapGeneratorP, self).__init__()
        self.block1 = ATP_Block(16, in_channel, 32, dropout=dropout)
        self.block2 = ATP_Block(16, 32, 64, dropout=dropout)
        self.block3 = ATP_Block(16, 64, 128, dropout=dropout)
        self.proj = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(4,4),stride=(4,4))
        self.weight = nn.Conv2d(dims[1], class_num, (1, 1), (1, 1))
        self.pool = nn.AdaptiveMaxPool2d(pool_size)

    def forward(self, image):
        x = self.block1(image)
        x = self.block2(x)
        x = self.block3(x)
        x = self.weight(x)
        x = self.pool(x)
        return x, self.weight.weight.squeeze() #[classnum, dims[1]]

class FPB_Extractor(nn.Module):
    def __init__(self, in_channel,hidden=512,pool_size=7,dropout=0.5):
        super(FPB_Extractor, self).__init__()
        self.block1 = ATP_Block(1,in_channel,32,dropout=dropout)
        self.block2 = ATP_Block(1, 32, 64, dropout=dropout)
        self.block3 = ATP_Block(1, 64, 128, dropout=dropout)
        self.block4 = ATP_Block(1, 128, 256, dropout=dropout)
        self.block5 = ATP_Block(1, 256, 512, dropout=dropout)
        self.proj = nn.Conv2d(in_channels=512,out_channels=hidden,kernel_size=[1,1],stride=[1,1])
        self.pool = nn.AdaptiveMaxPool2d(pool_size)

    def forward(self,image):
        x = self.block1(image)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)
        return x