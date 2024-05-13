import torch
import torch.nn as nn
import torchvision

import math

from models.FPN import FPN
from models.SCN import SCN,SCN_weak,SCN_not_group
from models.transformer_layers import SelfAttnLayer
from models.backbone import Backbone_split_vgg16,Groupwise_vgg16
from models.utils import weights_init


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

class SSNet(nn.Module):
    def __init__(self, in_channels, class_num,
                 spec_feat_layers = 1, spat_feat_layers=1,
                 spec_classify_layers = 1, spat_classify_layers = 1, merge_classify_layers=1 ,heads = 4,
                 hidden=512, dropout=0.5):
        super(SSNet, self).__init__()
        self.in_channels = in_channels
        self.class_num = class_num
        self.label_emb = nn.Embedding(class_num, hidden)
        self.label_emb.apply(weights_init)
        self.spatial_extractor = Backbone_split_vgg16(in_channel=16, dropout=dropout)#FPN(in_channels,hidden,dropout,7)
        # self.spectral_extractor = SCN(in_channels,hidden,dropout)#SCN(in_channels,hidden,dropout) #Groupwise_vgg16(in_channels=16, hidden=hidden)
        # self.spectral_extractor = SCN_weak(in_channels, hidden)
        self.spectral_extractor = SCN(in_channels, hidden, dropout)
        self.spat_emb_proj = nn.Linear(hidden,hidden)
        self.spec_emb_proj = nn.Linear(hidden,hidden)
        self.label_emb_ln = nn.LayerNorm(hidden)
        self.label_emb_map = nn.Sequential(
            nn.Linear(hidden,hidden),
            nn.GELU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden,hidden)
        )
        self.spat_feat_ln = nn.LayerNorm(hidden)
        self.spec_feat_ln = nn.LayerNorm(hidden)
        # self.spec_rep_map= nn.Sequential(
        #     nn.Linear(hidden,hidden),
        #     nn.GELU(),
        #     #nn.Dropout(dropout),
        #     nn.Linear(hidden,hidden)
        # )
        self.spat_classify_ln = nn.LayerNorm(hidden)
        self.spec_classify_ln = nn.LayerNorm(hidden)
        self.merge_classify_ln = nn.LayerNorm(hidden)
        self.spat_feat_trans = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(spat_feat_layers)])
        self.spec_feat_trans = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(spec_feat_layers)])
        self.spat_classify_trans = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(spat_classify_layers)])
        self.spec_classify_trans = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(spec_classify_layers)])
        self.merge_classify_trans = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(merge_classify_layers)])
        self.spat_feat_proj = nn.Linear(hidden,hidden)
        self.spec_feat_proj = nn.Linear(hidden,hidden)
        self.spat_linear = GroupWiseLinear(class_num,hidden)
        self.spec_linear = GroupWiseLinear(class_num,hidden)
        self.overall_linear = GroupWiseLinear(class_num, hidden)

        self.vote_weight = nn.Parameter(torch.Tensor(3,class_num), requires_grad=True)
        self.vote_weight.data.normal_(0.33333,1e-9)


    def forward(self, img):
        spatial_features = self.spatial_extractor(img)
        B, C, W, H = spatial_features.shape

        """空间特征"""
        spatial_tokens = spatial_features.view(B, C, -1).transpose(1, 2) #空间特征

        """通道特征"""
        spectral_splits, spectral_fuses = self.spectral_extractor(img)
        spectral_fuses = spectral_fuses.view(B,1,-1)
        # 下面三行本来是用来处理通道表示的
        #spectral_tokens = torch.cat(spectral_splits, dim=2).squeeze().transpose(1, 2)
        #spectral_tokens = self.spec_rep_map(spectral_tokens)
        #spectral_tokens = torch.cat([spectral_tokens, spectral_fuses],dim=1)
        spectral_tokens = spectral_fuses #[64,1,512]

        """标签嵌入映射"""
        label_emb = self.label_emb.weight
        label_emb = self.label_emb_ln(self.label_emb_map(label_emb))
        spat_emb = label_emb#self.spat_emb_proj(label_emb) + label_emb
        spec_emb = label_emb#self.spec_emb_proj(label_emb) + label_emb

        """transformer融合"""
        """空间特征提取"""
        spatial_tokens = self.spat_feat_ln(spatial_tokens)
        attns = []
        for layer in self.spat_feat_trans:
            spatial_tokens, attn = layer(spatial_tokens, mask=None)
            attns += attn.detach().unsqueeze(0).data

        """空间特征查询"""
        spatial_tokens_and_emb = torch.cat((spatial_tokens, spat_emb.repeat(B, 1, 1)), dim=1)
        spatial_tokens_and_emb = self.spat_classify_ln(spatial_tokens_and_emb)
        attns = []
        for layer in self.spat_classify_trans:
            spatial_tokens_and_emb, attn = layer(spatial_tokens_and_emb,mask=None)
            attns += attn.detach().unsqueeze(0).data

        spat_out_seq = spatial_tokens_and_emb[:, -self.class_num:, :]

        """通道特征提取"""
        spectral_tokens = self.spec_feat_ln(spectral_tokens)
        attns = []
        for layer in self.spec_feat_trans:
            spectral_tokens, attn = layer(spectral_tokens, mask=None)
            attns += attn.detach().unsqueeze(0).data

        """通道特征查询"""
        spectral_tokens_and_emb = torch.cat((spectral_tokens, spec_emb.repeat(B, 1, 1)), dim=1)
        spectral_tokens_and_emb = self.spec_classify_ln(spectral_tokens_and_emb)
        attns = []
        for layer in self.spec_classify_trans:
            spectral_tokens_and_emb, attn = layer(spectral_tokens_and_emb, mask=None)
            attns += attn.detach().unsqueeze(0).data


        spec_out_seq = spectral_tokens_and_emb[:, -self.class_num:, :]

        """双支融合"""
        # print(enhanced_spat_token.shape,spatial_tokens.shape,enhanced_spec_token.shape,spectral_tokens.shape)
        spatial_tokens = self.spat_feat_proj(spatial_tokens) + spatial_tokens
        spectral_tokens = self.spec_feat_proj(spectral_tokens) + spectral_tokens
        tokens = torch.cat((spatial_tokens, spectral_tokens, label_emb.repeat(B,1,1)), dim=1)
        tokens = self.merge_classify_ln(tokens)
        attns = []
        for layer in self.merge_classify_trans:
            tokens, attn = layer(tokens, mask=None)
            attns += attn.detach().unsqueeze(0).data

        out_seq = tokens[:, -self.class_num:, :]

        spat_logits = self.spat_linear(spat_out_seq)
        spec_logits = self.spec_linear(spec_out_seq)
        overall_logits = self.overall_linear(out_seq)

        # ensemble_logits = (spat_logits+spec_logits+overall_logits)/3
        vw = torch.softmax(self.vote_weight, dim=0)
        # ensemble_logits = vw[0]*spat_logits.detach() + vw[1]*spec_logits.detach() + vw[2]*overall_logits.detach()
        # ensemble_logits = vw[0] * spat_logits + vw[1] * spec_logits + vw[2] * overall_logits
        # ensemble_logits = spat_logits
        # ensemble_logits = spec_logits
        ensemble_logits = overall_logits

        return label_emb, spat_logits.detach(), spec_logits.detach(), overall_logits.detach(), ensemble_logits

        # return label_emb, spat_logits, spec_logits, overall_logits, ensemble_logits

class SSNet_doubleVGG(nn.Module):
    def __init__(self, in_channels, class_num,
                 spec_feat_layers = 1, spat_feat_layers=1,
                 spec_classify_layers = 1, spat_classify_layers = 1, merge_classify_layers=1 ,heads = 4,
                 hidden=512, dropout=0.5):
        super(SSNet_doubleVGG, self).__init__()
        self.hidden = hidden
        self.in_channels = in_channels
        self.class_num = class_num
        self.label_emb = nn.Embedding(class_num, hidden)
        self.label_emb.apply(weights_init)
        self.spatial_extractor = Backbone_split_vgg16(in_channel=16, dropout=dropout)#FPN(in_channels,hidden,dropout,7)
        self.spectral_extractor = Backbone_split_vgg16(in_channel=16, dropout=dropout)#SCN(in_channels,hidden,dropout) #Groupwise_vgg16(in_channels=16, hidden=hidden)
        # self.spectral_extractor = SCN_weak(in_channels, hidden)
        # self.spectral_extractor = SCN_not_group(in_channels, hidden, dropout)
        self.spat_emb_proj = nn.Linear(hidden,hidden)
        self.spec_emb_proj = nn.Linear(hidden,hidden)
        self.label_emb_ln = nn.LayerNorm(hidden)
        self.label_emb_map = nn.Sequential(
            nn.Linear(hidden,hidden),
            nn.GELU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden,hidden)
        )
        self.spat_feat_ln = nn.LayerNorm(hidden)
        self.spec_feat_ln = nn.LayerNorm(hidden)
        # self.spec_rep_map= nn.Sequential(
        #     nn.Linear(hidden,hidden),
        #     nn.GELU(),
        #     #nn.Dropout(dropout),
        #     nn.Linear(hidden,hidden)
        # )
        self.spat_classify_ln = nn.LayerNorm(hidden)
        self.spec_classify_ln = nn.LayerNorm(hidden)
        self.merge_classify_ln = nn.LayerNorm(hidden)
        self.spat_feat_trans = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(spat_feat_layers)])
        self.spec_feat_trans = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(spec_feat_layers)])
        self.spat_classify_trans = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(spat_classify_layers)])
        self.spec_classify_trans = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(spec_classify_layers)])
        self.merge_classify_trans = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(merge_classify_layers)])
        self.spat_feat_proj = nn.Linear(hidden,hidden)
        self.spec_feat_proj = nn.Linear(hidden,hidden)
        self.spat_linear = GroupWiseLinear(class_num,hidden)
        self.spec_linear = GroupWiseLinear(class_num,hidden)
        self.overall_linear = GroupWiseLinear(class_num, hidden)

        self.vote_weight = nn.Parameter(torch.Tensor(3,class_num), requires_grad=True)
        self.vote_weight.data.normal_(0.33333,1e-9)


    def forward(self, img):
        spatial_features = self.spatial_extractor(img)
        B, C, W, H = spatial_features.shape

        """空间特征"""
        spatial_tokens = spatial_features.view(B, C, -1).transpose(1, 2) #空间特征

        """通道特征"""
        spectral_fuses = self.spectral_extractor(img)
        spectral_fuses = spectral_fuses.view(B,-1,self.hidden)
        # 下面三行本来是用来处理通道表示的
        #spectral_tokens = torch.cat(spectral_splits, dim=2).squeeze().transpose(1, 2)
        #spectral_tokens = self.spec_rep_map(spectral_tokens)
        #spectral_tokens = torch.cat([spectral_tokens, spectral_fuses],dim=1)
        spectral_tokens = spectral_fuses #[64,1,512]

        """标签嵌入映射"""
        label_emb = self.label_emb.weight
        label_emb = self.label_emb_ln(self.label_emb_map(label_emb))
        spat_emb = label_emb#self.spat_emb_proj(label_emb) + label_emb
        spec_emb = label_emb#self.spec_emb_proj(label_emb) + label_emb

        """transformer融合"""
        """空间特征提取"""
        spatial_tokens = self.spat_feat_ln(spatial_tokens)
        attns = []
        for layer in self.spat_feat_trans:
            spatial_tokens, attn = layer(spatial_tokens, mask=None)
            attns += attn.detach().unsqueeze(0).data

        """空间特征查询"""
        spatial_tokens_and_emb = torch.cat((spatial_tokens, spat_emb.repeat(B, 1, 1)), dim=1)
        spatial_tokens_and_emb = self.spat_classify_ln(spatial_tokens_and_emb)
        attns = []
        for layer in self.spat_classify_trans:
            spatial_tokens_and_emb, attn = layer(spatial_tokens_and_emb,mask=None)
            attns += attn.detach().unsqueeze(0).data

        spat_out_seq = spatial_tokens_and_emb[:, -self.class_num:, :]

        """通道特征提取"""
        spectral_tokens = self.spec_feat_ln(spectral_tokens)
        attns = []
        for layer in self.spec_feat_trans:
            spectral_tokens, attn = layer(spectral_tokens, mask=None)
            attns += attn.detach().unsqueeze(0).data

        """通道特征查询"""
        spectral_tokens_and_emb = torch.cat((spectral_tokens, spec_emb.repeat(B, 1, 1)), dim=1)
        spectral_tokens_and_emb = self.spec_classify_ln(spectral_tokens_and_emb)
        attns = []
        for layer in self.spec_classify_trans:
            spectral_tokens_and_emb, attn = layer(spectral_tokens_and_emb, mask=None)
            attns += attn.detach().unsqueeze(0).data


        spec_out_seq = spectral_tokens_and_emb[:, -self.class_num:, :]

        """双支融合"""
        # print(enhanced_spat_token.shape,spatial_tokens.shape,enhanced_spec_token.shape,spectral_tokens.shape)
        spatial_tokens = self.spat_feat_proj(spatial_tokens) + spatial_tokens
        spectral_tokens = self.spec_feat_proj(spectral_tokens) + spectral_tokens
        tokens = torch.cat((spatial_tokens, spectral_tokens, label_emb.repeat(B,1,1)), dim=1)
        tokens = self.merge_classify_ln(tokens)
        attns = []
        for layer in self.merge_classify_trans:
            tokens, attn = layer(tokens, mask=None)
            attns += attn.detach().unsqueeze(0).data

        out_seq = tokens[:, -self.class_num:, :]

        spat_logits = self.spat_linear(spat_out_seq)
        spec_logits = self.spec_linear(spec_out_seq)
        overall_logits = self.overall_linear(out_seq)

        # ensemble_logits = (spat_logits+spec_logits+overall_logits)/3
        vw = torch.softmax(self.vote_weight, dim=0)
        # ensemble_logits = vw[0]*spat_logits.detach() + vw[1]*spec_logits.detach() + vw[2]*overall_logits.detach()
        ensemble_logits = vw[0] * spat_logits + vw[1] * spec_logits + vw[2] * overall_logits
        # ensemble_logits = spat_logits
        # ensemble_logits = spec_logits
        # ensemble_logits = overall_logits

        return label_emb, spat_logits, spec_logits, overall_logits, ensemble_logits


if __name__ == '__main__':
    img = torch.randn(64, 16, 256, 256)
    net = SSNet(16, 17)
    label_emb, spat_logits, spec_logits, out_logits = net(img)
    print(label_emb.shape, spat_logits.shape, spec_logits.shape, out_logits.shape)
