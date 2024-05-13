import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2
import math
from pdb import set_trace as stop

from .backbone import Backbone_split_res18,CNN,Backbone_split_vgg16
from .utils import custom_replace,weights_init
from .transformer_layers import SelfAttnLayer
from .FPN import FPN
from .FPB import ATP_Block,FPB_Extractor


#from .transformer_layers import SelfAttnLayer
#from .backbone import Backbone,Backbone_split_res101,Backbone_split_res34,Backbone_split_res18,Backbone_split_res50,CNN
#from .utils import custom_replace,weights_init
#from .position_enc import PositionEmbeddingSine,positionalencoding2d
#from .swin_transformer import SwinTransformer
#from .swin_transformer_v2 import SwinTransformerV2
#from .transformer_layers import SelfAttnLayer
#from .utils import custom_replace,weights_init
#from models.backbone import Backbone_split_res18

class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class GroupWiseConvBlock(nn.Module):
    def __init__(self,in_channel,output_dim,group_num,dropout=0.5):
        super(GroupWiseConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channel,
                                     out_channels=output_dim,
                                     kernel_size=(3,3),
                                     stride=(1,1),
                                     padding=(1,1),
                                     groups=group_num)
        self.norm1 = torch.nn.BatchNorm2d(output_dim)
        self.act = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(output_dim,output_dim,(3,3),(1,1),(1,1),groups=group_num)
        self.norm2 = torch.nn.BatchNorm2d(output_dim)

    def forward(self,x):
        x_ = self.conv1(x)
        x_ = self.norm1(x_)
        x_ = self.act(x_)
        x_ = self.dropout(x_)
        x_ = self.conv2(x_)
        x_ = self.norm2(x_) + x
        return x_

class GroupWiseConv(nn.Module):
    def __init__(self, in_channel:int, embed_dim=[16,128,512]):
        super(GroupWiseConv, self).__init__()
        self.groupWiseConvBlocks = torch.nn.ModuleList(
            [GroupWiseConvBlock(embed_dim[i+1],embed_dim[i+1],in_channel)
             for i in range(2)])
        self.lifters = torch.nn.ModuleList(
            [torch.nn.Conv2d(embed_dim[i],embed_dim[i+1],(3,3),(1,1),(1,1),groups=in_channel)  for i in range(2)]
        )
        self.norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm2d(embed_dim[i+1]) for i in range(2)]
        )
        self.acts = torch.nn.ModuleList(
            [torch.nn.GELU() for _ in range(2)]
        )
        self.dropouts = torch.nn.ModuleList(
            [torch.nn.Dropout(0.2) for _ in range(2)]
        )
        self.pools = torch.nn.ModuleList(
            [torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) for _ in range(1)]
        )
        self.final_pool = torch.nn.AdaptiveAvgPool2d(7)


    def forward(self,x):
        for i in range(2):
            x = self.lifters[i](x)
            x = self.norms[i](x)
            x = self.groupWiseConvBlocks[i](x)
            x = self.acts[i](x)
            if i != 2-1:
                x = self.dropouts[i](x)
                x = self.pools[i](x)
        x = self.final_pool(x)
        return x

class AttChannel(nn.Module):

    def __init__(self,args, in_channel, class_num, hidden=512,dropout=0.5):
        super(AttChannel, self).__init__()
        self.args = args
        self.class_num = class_num
        self.two_phase_start = args.two_phase_start
        #self.backbone = Backbone_split_vgg16(16) #Backbone_split_vgg16(16)#CNN(16)
        self.backbone = FPN(16, hidden, dropout, 7) #FPB_Extractor(in_channel,hidden)
        self.attMapGenerater = AttMapGenerator(in_channel=16,class_num=17,dropout=dropout)
        self.MLP = torch.nn.Sequential(
            torch.nn.Conv2d(hidden,64,(1,1),(1,1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(64,class_num,(1,1),(1,1)),
            torch.nn.BatchNorm2d(class_num),
            torch.nn.Sigmoid(),
        )
        trans_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden,nhead=4,dim_feedforward=1024,
                                                               norm_first=True,batch_first=True)
        trans_decoder_layer = torch.nn.TransformerDecoderLayer(d_model=hidden,nhead=4,dim_feedforward=1024,
                                                               norm_first=True,batch_first=True)
        norm = torch.nn.LayerNorm(hidden) #?
        self.trans_en = torch.nn.TransformerEncoder(encoder_layer=trans_encoder_layer,num_layers=args.layers,norm=norm)
        self.trans_de = torch.nn.TransformerDecoder(decoder_layer=trans_decoder_layer,num_layers=args.layers,norm=norm)
        self.pre_LN = torch.nn.LayerNorm(hidden)
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden, args.heads, dropout) for _ in range(args.layers)])

        self.label_embedding = torch.nn.Embedding(class_num,hidden)
        self.month_embedding = torch.nn.Embedding(12, hidden, padding_idx=None)
        self.loc_embedding = torch.nn.Embedding(11 * 26, hidden, padding_idx=None)

        self.feature_pool = torch.nn.AdaptiveAvgPool2d(7)
        self.group_wise_linear = GroupWiseLinear(num_class=class_num,hidden_dim=hidden)
        self.classifier1 = torch.nn.Sequential(
            torch.nn.Linear(hidden * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            torch.nn.Linear(4096, class_num),
        )
        #self.vote_weight = torch.nn.parameter.Parameter(torch.Tensor(1,2,class_num), requires_grad=True)
        #self.init_vote_weight()

        self.featSqueeze = TokenSqueeze(hidden,0.2)
        self.embSqueeze = TokenSqueeze(hidden, 0.2)
        self.monthSqueeze = TokenSqueeze(hidden,0.2)
        self.locSqueeze = TokenSqueeze(hidden,0.2)

        self.voter = torch.nn.Linear(2,1)

        self.month_embedding.apply(weights_init)
        self.label_embedding.apply(weights_init)
        self.loc_embedding.apply(weights_init)

    def forward_token_gen(self,att_map,features):
        """
        :param chan_features: [b,hidden,w,h]
        :param features: [b,hidden,w,h]
        :return:tokens: [b,c,hidden]
        """
        shp = att_map.shape
        att_map_f = att_map.view(shp[0],shp[1],-1)

        shp = features.shape
        #features_f = chan_features.view(shp[0],shp[1],-1).transpose(1,2)
        features_f = features.view(shp[0],shp[1],-1).transpose(1,2)
        tokens = torch.matmul(att_map_f,features_f) #[b,c,hidden]

        return tokens

    def forward(self, x, epoch, month_num, loc_num):

        features = self.feature_pool(self.backbone(x))  # [b,hidden,w,h]
        chan_features, attWeight = self.attMapGenerater(x)  # [b,hidden,w,h]
        tokens = self.forward_token_gen(chan_features, features)  # [b,c,hidden]

        """
            similar restrain
        """
        eyeTarget = torch.eye(self.class_num, requires_grad=False)
        similarMatrix = torch.matmul(attWeight,attWeight.transpose(0,1))
        # similarMatrix = torch.softmax(similarMatrix,dim=0)
        #print(similarMatrix)
        corr_matrix = (((eyeTarget.cuda()-similarMatrix)**2)*torch.abs(1-eyeTarget).cuda())
        similar_loss = corr_matrix.sum()
        #print(similar_loss)

        B, C, W, H = features.shape
        features_f = features.view(B, C, -1).transpose(1, 2).contiguous()

        B = tokens.shape[0]
        # print(features_f.shape,tokens.shape,self.label_embedding.weight.shape)
        l_emb = self.label_embedding.weight.unsqueeze(0).repeat(B, 1, 1)
        # l_emb = self.embSqueeze(l_emb)

        if self.args.use_month:
            month_embedding = self.month_embedding(month_num.long().cuda())
            # print(month)
            # month_out = self.month_mlp(month_embedding)
            month_embedding = month_embedding.view(month_embedding.shape[0], 1, -1)
            month_embedding = month_embedding.repeat(1, l_emb.shape[1], 1)
            # zmonth_embedding = self.monthSqueeze(month_embedding)
            l_emb = l_emb + month_embedding

        if self.args.use_loc:
            loc_embedding = self.loc_embedding(loc_num.long().cuda())
            # print(month)
            # loc_out = self.month_mlp(month_embedding)
            loc_embedding = loc_embedding.view(loc_embedding.shape[0], 1, -1)
            loc_embedding = loc_embedding.repeat(1, l_emb.shape[1], 1)
            # loc_embedding = self.locSqueeze(loc_embedding)
            l_emb = l_emb + loc_embedding

        #tokens = self.featSqueeze(tokens)
        token_and_embeddings = torch.cat((tokens, l_emb), dim=1)
        #token_and_embeddings = tokens
        aug_tokens = self.trans_en(tokens)[:, -self.class_num:, :]  # [b,c,hidden]
        de_output = self.trans_de(l_emb,aug_tokens)

        token_and_embeddings = self.pre_LN(token_and_embeddings)
        attns = []
        for layer in self.self_attn_layers:
            token_and_embeddings, attn = layer(token_and_embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data
            #print(attn.shape)
        aug_tokens = token_and_embeddings[:,-self.class_num:,:]


        # representation = self.feature_pool(features)  # [b,1,hidden]
        shp = features.shape
        # print(representation.shape)
        conv_out = self.classifier1(features.view(shp[0], -1))
        trans_out = self.group_wise_linear(aug_tokens)
        shp = conv_out.shape
        conv_out = conv_out.view(shp[0], 1, -1)
        shp = trans_out.shape
        trans_out = conv_out.view(shp[0], 1, -1)
        candidates = torch.cat((conv_out, trans_out), dim=1)  # [b,2,c]
        # print(candidates.shape, self.vote_weight.shape)
        # out = torch.mean(candidates, dim=1)
        #out,_ = torch.max(candidates, dim=1)
        #print(candidates.shape)
        #out = self.voter(candidates.transpose(1,2)).squeeze()
        #vote_weight_norm = torch.softmax(self.vote_weight,dim=1)
        #print("")
        #print(self.vote_weight[0,:,0],vote_weight_norm[0,:,0])
        #self.vote_weight.requires_grad=True
        #out = (vote_weight_norm*candidates).mean(dim=1)

        return candidates,similar_loss #candidates[:,1,:]

class AttChannelPacked(nn.Module):
    def __init__(self,args, in_channel, class_num, hidden=512,dropout=0.5):
        super(AttChannelPacked, self).__init__()
        self.args = args
        self.ac = AttChannel(args,in_channel,class_num,hidden,dropout)
        self.vote_weight = torch.nn.parameter.Parameter(torch.ones(1, 2, class_num)*0.5, requires_grad=True) # torch.Tensor(1, 2, class_num)
        self.vote_weight_freeze = torch.nn.parameter.Parameter(torch.Tensor(1, 2, class_num), requires_grad=False)
        self.init_vote_weight_random(self.vote_weight)

    def init_vote_weight_constant(self,parameter):
        # stdv = 1. / math.sqrt(self.vote_weight.size(0))
        # for i in range(2):
        #     for j in range(self.class_num):
        #         self.vote_weight[0][i][j].data.uniform_(-stdv, stdv)
        torch.nn.init.constant_(parameter,0.5)
        nn.init.normal_(self.vote_weight,0.5,0.05)

    def init_vote_weight_random(self,parameter):
        # stdv = 1. / math.sqrt(self.vote_weight.size(0))
        # for i in range(2):
        #     for j in range(self.class_num):
        #         self.vote_weight[0][i][j].data.uniform_(-stdv, stdv)
        #torch.nn.init.constant_(parameter,0.5)
        stdv = 1. / math.sqrt(self.vote_weight.size(1)*self.vote_weight.size(0))
        nn.init.normal_(self.vote_weight,0.5,0.0005)
        # print(self.vote_weight[0,:,0])

    def forward(self, x, epoch, month_num, loc_num):
        if(epoch <= self.args.two_phase_start):
            candidates,similar_loss = self.ac(x,epoch,month_num,loc_num)
            vote_weight_norm = torch.softmax(self.vote_weight, dim=1)
            #print(vote_weight_norm)
            out = candidates.mean(dim=1)
            # out = candidates[:,1,:]
            # with torch.no_grad():
            #     vote_weight_norm = torch.softmax(self.vote_weight, dim=1)
            #     out = (vote_weight_norm*candidates).sum(dim=1)
        else:
            with torch.no_grad():
                candidates, similar_loss = self.ac(x, epoch, month_num, loc_num)
            vote_weight_norm = torch.softmax(self.vote_weight, dim=1)
            out = (vote_weight_norm * candidates).sum(dim=1)

        return out, similar_loss, candidates

class TokenSqueeze(nn.Module):
    def __init__(self, feature_size, dropout):
        super().__init__()
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(feature_size, (int)(feature_size/4)),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear((int)(feature_size/4),feature_size),
        )

    def forward(self, tokens):
        return self.MLP(tokens)

class AttMapGenerator(nn.Module):
    def __init__(self, in_channel,class_num,dims=[64,128],pool_size=7,dropout=0.5):
        super(AttMapGenerator, self).__init__()
        self.featureCap = nn.Sequential(nn.Conv2d(in_channel,dims[0],(4,4),(4,4),groups=16),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout,inplace=True),
                                 nn.Conv2d(dims[0],dims[1],(2,2),(2,2),groups=16),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout, inplace=True))
        self.weight = nn.Conv2d(dims[1],class_num,(1,1),(1,1))
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

    def forward(self, image):
        x = self.featureCap(image)
        x = self.weight(x)
        x = self.pool(x)
        return x, self.weight.weight.squeeze() #[classnum, dims[1]]





if __name__ == '__main__':
    groupWiseConv = GroupWiseConv(16)
    x = torch.randn(64,16,400,400)
    print(groupWiseConv(x).shape)
