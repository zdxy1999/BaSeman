import torch
import torch.nn as nn
import torchvision

class NoneLinearLayer(nn.Module):
    def __init__(self, in_channels, hidden, groups, firstKernelSize):
        super(NoneLinearLayer, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, hidden, groups=1, kernel_size=(firstKernelSize, firstKernelSize),padding=firstKernelSize // 2)
        self.conv1 = nn.Conv2d(in_channels,hidden, groups=groups, kernel_size=(firstKernelSize,firstKernelSize), padding = firstKernelSize//2)
        self.act = nn.LeakyReLU(inplace=True)#nn.GELU() #nn.LeakyReLU()
        # self.conv2 = nn.Conv2d(hidden, in_channels, groups=1, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(hidden,in_channels, groups=groups, kernel_size=(3,3), padding=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class SCN_Block(nn.Module):

    def __init__(self, in_channels, out_channels, groups, dropout):
        super(SCN_Block, self).__init__()

        self.split_conv = nn.Conv2d(in_channels, out_channels, groups=groups, kernel_size=(4,4), stride=(4,4)) #有16个通道 要*16
        #self.split_conv = nn.Conv2d(in_channels, out_channels, groups=1, kernel_size=(4, 4),stride=(4, 4))  # 有16个通道 要*16
        self.nonLinLayer1 = NoneLinearLayer(out_channels, out_channels*2, groups,7)
        self.nonLinLayer2 = NoneLinearLayer(out_channels, out_channels*2, groups,5)
        self.nonLinLayer3 = NoneLinearLayer(out_channels, out_channels*2, groups,3)
        self.nonLinLayer4 = NoneLinearLayer(out_channels, out_channels * 2, groups, 3)
        fuse_and_split_num = (in_channels//groups) * (groups + 1)
        self.fuse_conv = nn.Conv2d(fuse_and_split_num, out_channels//groups, groups=1, kernel_size=(4,4),stride=(4,4))
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels//groups)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act1 = nn.LeakyReLU(inplace=True) #nn.ReLU(inplace=True)#nn.LeakyReLU()
        self.act2 = nn.LeakyReLU(inplace=True)#nn.LeakyReLU()

    
    def forward(self,split,fused):
        #split_f = self.split_conv0(split)
        split_f = self.split_conv(split)
        # 通道特征与全局特征融合

        # split_f = self.dropout1(split_f)
        # fused_f = self.dropout2(fused_f)
        # split_f = self.bn1(split_f)
        split_f = self.act1(split_f)
        # fused_f = self.bn2(fused_f)


        # 后面加的残差块
        split_f = split_f + self.nonLinLayer1(split_f)
        split_f = split_f + self.nonLinLayer2(split_f)
        split_f = split_f + self.nonLinLayer3(split_f)
        split_f = split_f + self.nonLinLayer4(split_f)
        #split_f = split_f + self.nonLinLayer3(split_f)

        fused_f = self.fuse_conv(torch.cat((split, fused), dim=1)) #移到后面
        fused_f = self.act2(fused_f)

        return split_f,fused_f

class SCN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(SCN, self).__init__()
        self.in_channels = in_channels
        self.proj_1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(1,1), stride=(1,1))
        self.block1 = SCN_Block(in_channels=1 * 16, out_channels=4 * 16, groups=in_channels, dropout=dropout)
        self.block2 = SCN_Block(in_channels=4 * 16, out_channels=16 * 16, groups=in_channels, dropout=dropout)
        self.block3 = SCN_Block(in_channels=16 * 16, out_channels=64 * 16, groups=in_channels, dropout=dropout)
        self.block4 = SCN_Block(in_channels=64 * 16, out_channels=256 * 16, groups=in_channels, dropout=dropout)
        self.proj_split = nn.Conv2d(in_channels=256 * 16, out_channels=out_channels * 16, kernel_size=(1,1), stride=(1,1), groups=in_channels)
        #self.proj_fused = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), groups=1)
        self.proj_fused = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=(1,1), stride=(1,1), groups=in_channels)

    def forward(self, img):
        split = img
        fused = self.proj_1(img)

        split, fused = self.block1(split, fused)
        split, fused = self.block2(split, fused)
        split, fused = self.block3(split, fused)
        split, fused = self.block4(split, fused)
        split = self.proj_split(split)
        fused = self.proj_fused(fused) # 全局的各通道融合特征
        channel_fs = []
        for i in range(self.in_channels): # 各通道的特征
            d = split.shape[1]//16
            channel_fs.append(split[:, i*d:(i+1)*d, :, :])

        return channel_fs, fused


class SCN_Block_weak(nn.Module):

    def __init__(self, in_channels, out_channels, groups, dropout):
        super(SCN_Block_weak, self).__init__()

        #self.split_conv = nn.Conv2d(in_channels, out_channels, groups=groups, kernel_size=(4, 4),stride=(4, 4))  # 有16个通道 要*16
        self.split_conv = nn.Conv2d(in_channels, out_channels, groups=1, kernel_size=(4, 4),stride=(4, 4))  # 有16个通道 要*16
        self.nonLinLayer1 = NoneLinearLayer(out_channels, out_channels * 2, groups, 7)
        self.nonLinLayer2 = NoneLinearLayer(out_channels, out_channels * 2, groups, 5)
        self.nonLinLayer3 = NoneLinearLayer(out_channels, out_channels * 2, groups, 3)
        self.nonLinLayer4 = NoneLinearLayer(out_channels, out_channels * 2, groups, 3)
        fuse_and_split_num = (in_channels // groups) * (groups + 1)
        self.fuse_conv = nn.Conv2d(fuse_and_split_num, out_channels // groups, groups=1, kernel_size=(4, 4),
                                   stride=(4, 4))
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels//groups)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act1 = nn.LeakyReLU(inplace=True)  # nn.ReLU(inplace=True)#nn.LeakyReLU()
        self.act2 = nn.LeakyReLU(inplace=True)  # nn.LeakyReLU()

    def forward(self, split, fused):
        # split_f = self.split_conv0(split)
        split_f = self.split_conv(split)
        # 通道特征与全局特征融合

        # split_f = self.dropout1(split_f)
        # fused_f = self.dropout2(fused_f)
        # split_f = self.bn1(split_f)
        split_f = self.act1(split_f)
        # fused_f = self.bn2(fused_f)

        # 后面加的残差块
        split_f = split_f + self.nonLinLayer1(split_f)
        split_f = split_f + self.nonLinLayer2(split_f)
        split_f = split_f + self.nonLinLayer3(split_f)
        split_f = split_f + self.nonLinLayer4(split_f)
        # split_f = split_f + self.nonLinLayer3(split_f)

        fused_f = self.fuse_conv(torch.cat((split, fused), dim=1))  # 移到后面
        fused_f = self.act2(fused_f)

        return split_f, fused_f

class SCN_weak(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCN_weak, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.pool = nn.AdaptiveMaxPool2d((1,1))

    def forward(self, img):
        fused = self.conv(img)
        fused = self.pool(fused)
        return None, fused

class SCN_not_group(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(SCN_not_group, self).__init__()
        self.in_channels = in_channels
        self.proj_1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(1,1), stride=(1,1))
        self.block1 = SCN_Block_weak(in_channels=1 * 16, out_channels=4 * 16, groups=16, dropout=dropout)
        self.block2 = SCN_Block_weak(in_channels=4 * 16, out_channels=16 * 16, groups=16, dropout=dropout)
        self.block3 = SCN_Block_weak(in_channels=16 * 16, out_channels=64 * 16, groups=16, dropout=dropout)
        self.block4 = SCN_Block_weak(in_channels=64 * 16, out_channels=256 * 16, groups=16, dropout=dropout)
        self.proj_split = nn.Conv2d(in_channels=256 * 16, out_channels=out_channels * 16, kernel_size=(1,1), stride=(1,1), groups=1)
        #self.proj_fused = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), groups=1)
        self.proj_fused = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=(1,1), stride=(1,1), groups=1)

    def forward(self, img):
        split = img
        fused = self.proj_1(img)

        split, fused = self.block1(split, fused)
        split, fused = self.block2(split, fused)
        split, fused = self.block3(split, fused)
        split, fused = self.block4(split, fused)
        split = self.proj_split(split)
        fused = self.proj_fused(fused) # 全局的各通道融合特征
        channel_fs = []
        for i in range(self.in_channels): # 各通道的特征
            d = split.shape[1]//16
            channel_fs.append(split[:, i*d:(i+1)*d, :, :])

        return channel_fs, fused

if __name__ == '__main__':
    model = SCN(16, 512, 0.5)
    img = torch.randn(64, 16, 256, 256)
    print(model(img)[0].shape, model(img)[1].shape)

