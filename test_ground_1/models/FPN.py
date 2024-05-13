import torch
import torch.nn as nn
import torchvision

class FPN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(FPN_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, groups=16)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, dilation=1, padding=1, groups=16)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, dilation=2, padding=2, groups=16)
        #self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, dilation=7, padding=7)
        ##self.fuser = nn.Conv2d(in_channels=3*out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1)
        self.weight = nn.parameter.Parameter(torch.Tensor(3), requires_grad=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(stride=2,kernel_size=(2,2))

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(x)
        f3 = self.conv3(x)
        # f4 = self.conv4(x)
        weight = torch.softmax(self.weight, dim=0)
        f = weight[0]*f1 + weight[1]*f2 + weight[2]*f3 #+ weight[3]*f4
        # f = self.bn(f)
        f = self.act(f)
        f = self.dropout(f)
        f = self.pool(f)

        return f

class FPN(nn.Module):
    def __init__(self, in_channels, hidden, dropout, pool_size):
        super(FPN, self).__init__()
        self.block1 = FPN_Block(in_channels,32,dropout=dropout)
        self.block2 = FPN_Block(32, 64, dropout=dropout)
        self.block3 = FPN_Block(64, 128, dropout=dropout)
        self.block4 = FPN_Block(128, 256, dropout=dropout)
        self.block5 = FPN_Block(256, 512, dropout=dropout)
        self.proj = nn.Conv2d(in_channels=512,out_channels=hidden,kernel_size=(1,1),stride=(1,1))
        self.pool = nn.AdaptiveMaxPool2d(pool_size)

    def forward(self, img):
        f = self.block1(img)
        f = self.block2(f)
        f = self.block3(f)
        f = self.block4(f)
        f = self.block5(f)
        f = self.proj(f)
        return self.pool(f)

if __name__ == '__main__':
    model = FPN(16,512,0.5,7)
    img = torch.randn(64,16,256,256)
