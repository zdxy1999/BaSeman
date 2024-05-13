import numpy as np
import pandas as pd
import torch
import lzma
import glob
import gzip
import argparse
import torchvision.models

import models.backbone

'''
attens = torch.load('./results/split_16c.cnn.LSCIDMR_16c.2layer.4heads.bsz_64.adam5e-05.plateau_on_map.use_month.use_loc.group_wise_linear_4group_lrelu/attens.pt')
print(attens[0].shape)
#plt.matshow(attens[0][21,:,:].detach().cpu().numpy())
names = ['TC','EC','FS','WJ','Snow','Ocean','Desert','Veg','Clr', 'Ci', 'Cs', 'DC', 'Ac', 'As', 'Ns', 'Cu', 'Sc', 'St', 'NK', 'ND']
matrix = pd.pivot_table(
    pd.DataFrame(attens[0][21,-17:,-17:].detach().cpu().numpy()),index=0,columns=1,aggfunc="size",fill_value=0
).values.tolist()
Chord(matrix,
      names,
      colors='d3.schemeSet2').show()
print(torchvision.models.vgg19())
#print(attens[0][0,:,:])
'''

'''
labels = torch.Tensor([0,0,0,0,0,0,0,0,  0.9,1,0.05,0.5,0.049,0,0,0,0])

labels[-9:][labels[-9:]<0.05] = torch.log(10*(labels[-9:][labels[-9:]<0.05]+1e-9)/\
                                                            (1-10*(labels[-9:][labels[-9:]<0.05]+1e-9)))
labels[-9:][labels[-9:] >= 0.05] = (1/1.9)*(labels[-9:][labels[-9:]>=0.05]+0.9)/\
                                                     (1-(1/1.9)*(labels[-9:][labels[-9:]>=0.05]+0.9))

def left_region_project(tensor):
    return torch.log(10*(tensor+1e-12)/(1-(10)*(tensor+1e-12)))

def right_region_project(tensor):
    return torch.log((1/1.9)*(tensor+0.9)/(1-(1/1.9)*(tensor+0.9)))

def reproject(tensor):
    tensor[tensor < 0.05] = left_region_project( tensor[tensor< 0.05])
    tensor[tensor >=0.05] = right_region_project(tensor[tensor>=0.05])
    return torch.sigmoid(tensor)




print(reproject(torch.Tensor([0,1,0.05,0.5,0.049])))
#print(torch.sigmoid(torch.Tensor(0.5))
'''

img = torch.randn((64,16,256,256))
net = models.backbone.Backbone_split_res18(16)
print(net)








