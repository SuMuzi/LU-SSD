import torch
import math
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from module.mamba2 import Mamba2_2,Mamba2_1
from module.mamba_util import bulid_act
from mamba_ssm import Mamba2
from module.mamba_util import Mlp,MLP_SCSSD,ConvFFN_SCSSD
from einops import rearrange
import pywt
import numpy as np
class IntensityClassifier(nn.Module):

    def __init__(self, in_c=1, K=5, base=64,act=None):
        super().__init__()
        self.body = nn.Sequential(
            # nn.AdaptiveAvgPool2d(8),           
            nn.Conv2d(in_c, base, 3, 1, 1),
            act,
            nn.Conv2d(base, base*2, 3, 1, 1),
            act,
            nn.Conv2d(base*2, K, 3, 1, 1),

            nn.Softmax(dim=1)  
        )

    def forward(self, x):
        return self.body(x)   # (B,K)

class Extract_high_freq_by_walvet(nn.Module):


    def __init__(self, wavelet='db4'):
        super().__init__()
        self.wavelet = wavelet

    def forward(self,data):
        coeffs = pywt.dwt2(data.cpu().numpy(),self.wavelet)
        cA,(cH,cV,cD) = coeffs 
        hf_codffs = (cA,(2 * cH,2 * cV,2 * cD))
        hf_img = pywt.idwt2(hf_codffs,self.wavelet)
        hf_img = torch.from_numpy(hf_img)
        return hf_img.to('cuda')

class HighPass(nn.Module):
    def __init__(self):
        super(HighPass, self).__init__()

        self.kernel = torch.tensor([[-1.,-1.,-1.],
                                    [-1.,8.,-1.],
                                    [-1.,-1.,-1.]],dtype=torch.float32)
        self.register_buffer('kernel',self.kernel.view(1,1,3,3))

    def forward(self,x):
        # x:(B,C,H,W)->高通残差
        return x-F.conv2d(x,self.kernel,padding=1)
class Upsample(nn.Sequential):

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))

        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))

        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
class Upsampler(nn.Sequential):


    def __init__(self, scale, num_feat,act):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2)),
                m.append(act)
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3)),
            m.append(act)
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsampler, self).__init__(*m)
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,act=None):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels,
                                                 in_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 groups=in_channels),  
                                       act)
                                       # nn.LeakyReLU(negative_slope=0.2,inplace=True))

        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                       act)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class data_geolsm_concate(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,act=None):
        super(data_geolsm_concate, self).__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels,
                                                 out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       act)


    def forward(self, x):
        x = self.depthwise(x)

        return x
class geo_lsm_Branch(nn.Module):
    def __init__(self, in_channels, out_channels, scale,kernel_size=2, stride=2, padding=0,act=None):
        super(geo_lsm_Branch, self).__init__()
        self.branch = nn.ModuleList()
        for _ in range(int(math.log(scale, stride))):
            self.branch.append(nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding), 
                act

            ))

        # nn.LeakyReLU(negative_slope=0.2,inplace=True))

    def forward(self, x):
        out = []
        for block in self.branch:
            x = block(x)
            out.append(x)
        return out

class GlobalSeed(nn.Module):
    def __init__(self, C, ssd_config, G=8, d_state=16, d_conv=3, expand=2, split_num=1):
        super().__init__()
        self.tokenize = nn.Conv2d(C, G, 1)          
        self.mamba = Mamba2_1(
            d_model = G,          # Model dimension d_model
            d_state = d_state,    # SSM state expansion factor
            d_conv = d_conv,      # Local convolution width
            expand = expand,      # Block expansion factor
            headdim = 8,
            **ssd_config
        )        
        self.detokenize = nn.Conv2d(G, C, 1)        

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.tokenize(x)                        # (B,G,H,W)
        g = rearrange(g, 'b g h w -> b (h w) g')
        g = self.mamba(g,H,W) + g                       
        g = rearrange(g, 'b (h w) g -> b g h w', h=H, w=W)
        return x + self.detokenize(g)   

class Cascade_SCSSD_Block(nn.Module):
    def __init__(self, in_channel_list,out_channel_list,ssd_config,split_nums,d_state=64, d_conv=3, expand=2):
        super(Cascade_SCSSD_Block, self).__init__()

        self.blocks = nn.ModuleList([Mamba2_2(d_model = in_channel_list[i] // split_nums,  # Model dimension d_model
                                              out_channel = out_channel_list[i] // split_nums,  # Model dimension d_model
                                              d_state = d_state,  # SSM state expansion factor
                                              d_conv = d_conv,  # Local convolution width
                                              expand = expand,  # Block expansion factor
                                              headdim = 8,
                                              **ssd_config
                                              )
                                     for i in range(len(in_channel_list))])
    def forward(self, x,h,w):
        for block in self.blocks:
            x = block(x,h,w)
        return x

class SubCrossToken(nn.Module):
    def __init__(self, split_num):
        super().__init__()
        self.split_num = split_num
        self.W = nn.Parameter(torch.eye(self.split_num) + 0.01 * torch.randn(self.split_num, self.split_num))
    def forward(self, xs):

        B, L, C = xs[0].shape
        g = [x.mean(dim=1) for x in xs]                              # (B,C/4)
        g = torch.stack(g, dim=2)                                    # (B,C/4,4)
        weight = F.softmax(self.W, dim=1)                            # (split_num,split_num)
        g = torch.einsum('bcq,qp->bcp', g, weight)                  
        g = [g[:, :, i].unsqueeze(1).expand(-1,L,-1) for i in range(self.split_num)]
        return [xs[i] + g[i] for i in range(self.split_num)]

class GlobalFusion(nn.Module):
    def __init__(self, C, G=8):
        super().__init__()
        self.low = nn.Conv2d(C, G, 1)
        self.high = nn.Conv2d(G, C, 1)

    def forward(self, x):
        return x + self.high(self.low(x))   

class SCSSD(nn.Module):
    def __init__(self, in_channel_list, out_channel_list,ssd_config,split_nums=8,d_state=64, d_conv=3, expand=2,mode='encoder'):
        super().__init__()
        self.input_dim = in_channel_list
        self.output_dim = out_channel_list
        self.split_nums = split_nums
        self.d_state = d_state
        self.ssd_config = ssd_config
        self.global_seed = GlobalSeed(self.input_dim[0],ssd_config)
        self.cascade_scssd = Cascade_SCSSD_Block(self.input_dim, self.output_dim,self.ssd_config,self.split_nums)
        self.sub_mamba = nn.ModuleList([self.cascade_scssd for _ in range(self.split_nums)])
        self.sub_cross = SubCrossToken(self.split_nums)
        self.norm_1 = nn.LayerNorm(self.input_dim[0])
        self.norm_2 = nn.LayerNorm(self.output_dim[-1])
        self.global_fusion = GlobalFusion(self.output_dim[-1])
        if mode == 'encoder':
            self.fnn = ConvFFN_SCSSD(self.input_dim[-1],int(2*self.output_dim[-1]))
        elif mode == 'decoder':
            self.fnn = ConvFFN_SCSSD(self.input_dim[-1],int(self.output_dim[-1]/2))
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):

        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        # 1) 预埋全局
        seed = self.global_seed(x)

        B, C = x.shape[:2]
        # assert C == self.input_dim

        n_tokens = x.shape[2] * x.shape[3]
        img_dims = x.shape[2:]
        # print("x.reshape(B, C, n_tokens): ",x.reshape(B, C, n_tokens).shape)
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # print("x_flat: ", x_flat.shape)

        x_norm = self.norm_1(x_flat)

        xs = [self.sub_mamba[i](x_i,int(math.sqrt(n_tokens)),int(math.sqrt(n_tokens))) + x_i * self.skip_scale
              for i, x_i in enumerate(torch.chunk(x_norm, self.split_nums, 2))]

        xs = self.sub_cross(xs)

        out = torch.cat(xs, dim=2)
        out = self.norm_2(out)
        out = out.transpose(-1, -2).reshape(B, self.output_dim[-1], *img_dims)

        out = out + seed
        # out = self.global_fusion(out)

        out = self.fnn(out)

        return out

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=8,act=None):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            act,
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.act = act

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.act(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self,act=None):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = act

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel, ssd_config, split_nums,act):
        super(CBAM, self).__init__()
        self.split_nums = split_nums
        self.ssd_config = ssd_config
        self.channel_attention = ChannelAttentionModule(channel,act=act)
        self.spatial_attention = SpatialAttentionModule(act=act)
        self.gs = nn.Sequential(nn.GroupNorm(self.split_nums,channel),
                                act)
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return self.gs(out)


class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, ssd_config, split_nums,act):
        super(MSCB, self).__init__()

        self.split_nums = split_nums
        self.ssd_config = ssd_config
        self.act = act

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                   self.act)

        self.dwcovn1 = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 groups=out_channels)  

        self.dwcovn2 = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=out_channels) 

        self.dwcovn3 = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2,
                                 groups=out_channels)  

        self.pointwise = nn.Sequential(nn.Conv2d(3 * out_channels,
                                                 in_channels,
                                                 kernel_size=1),
                                       self.act)

        self.cbam = CBAM(in_channels, self.ssd_config, self.split_nums,act=self.act)

    def forward(self, x):
        if self.ssd_config['MSCB'] and self.ssd_config['CBAM']:
            x_init = x
            x = self.conv1(x)
            x1 = self.dwcovn1(x)
            x2 = self.dwcovn2(x)
            x3 = self.dwcovn3(x)
            x = torch.cat((x1,x2,x3),dim=1)
            x = self.pointwise(x) + x_init
            x = self.act(x)
            x = self.cbam(x)
            return x

        elif self.ssd_config['MSCB'] and not self.ssd_config['CBAM']:
            x_init = x
            x = self.conv1(x)
            x1 = self.dwcovn1(x)
            x2 = self.dwcovn2(x)
            x3 = self.dwcovn3(x)
            x = torch.cat((x1,x2,x3),dim=1)
            x = self.pointwise(x) + x_init
            x = self.act(x)
            return x

        elif not self.ssd_config['MSCB'] and self.ssd_config['CBAM']:
            return self.cbam(x)
        else:
            return None



class LU_M2SR(nn.Module):

    def __init__(self,
                 input_channels=3,
                 out_channels=1,
                 rs_factor=2,
                 c_list=[],
                 res=True,
                 is_cls=False,
                 split_nums=8,
                 atten_config={},
                 ssd_config={},
                 dic_geo_lsm={},
                 act_name='silu',
                 norm_name='bn'):
        super().__init__()

        self.rs_factor = rs_factor
        self.c_list = c_list
        self.out_channels = out_channels
        self.input_channels = input_channels
        self.split_nums = split_nums
        self.atten_config = atten_config
        self.ssd_config = ssd_config
        self.dic_geo_lsm = dic_geo_lsm
        self.res = res
        self.is_cls = is_cls
        self.act_name = act_name
        self.norm_name = norm_name

        self.act,self.norm = bulid_act(self.act_name)

        self.wavelet_brand = Extract_high_freq_by_walvet()
        # encoder_stage1

        if self.dic_geo_lsm['geo_lsm']:
            self.geo_lsm = geo_lsm_Branch(1, 1,self.rs_factor, act=self.act)
            self.stage1_en = DepthwiseSeparableConv2d(input_channels,input_channels,act=self.act)
            # self.fusion = nn.Sequential(nn.Conv2d(input_channels+1,self.c_list[0],kernel_size=1,stride=1),
            #                             self.act)

        else:
            self.stage1_en = DepthwiseSeparableConv2d(input_channels, self.c_list[0],act=self.act)
        self.contacate = data_geolsm_concate(input_channels+1,self.c_list[0],act=self.act)
        self.att_1 = MSCB(self.c_list[0],3,self.atten_config,self.split_nums,act=self.act)
        # encoder_stage2
        self.stage2_en = SCSSD(in_channel_list=[self.c_list[0],self.c_list[0]],
                                     out_channel_list=[self.c_list[0],self.c_list[0]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode = 'encoder') 
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gs_2 = nn.Sequential(nn.GroupNorm(self.split_nums,self.c_list[1]),
                                 self.act)
        self.att_2 = MSCB(self.c_list[1],3,self.atten_config,self.split_nums,act=self.act)
        # encoder_stage3
        self.stage3_en = SCSSD(in_channel_list=[self.c_list[1],self.c_list[1]],
                                     out_channel_list=[self.c_list[1],self.c_list[1]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode='encoder'
                                     )#2层
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.att_3 = MSCB(self.c_list[2],3,self.atten_config,self.split_nums,act=self.act)
        self.gs_3 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[2]),
                                  self.act)
        # encoder_stage4
        self.stage4_en = SCSSD(in_channel_list=[self.c_list[2],self.c_list[2]],
                                     out_channel_list=[self.c_list[2],self.c_list[2]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode='encoder'
                                     )
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.att_4 = MSCB(self.c_list[3],3,self.atten_config,self.split_nums,act=self.act)
        self.gs_4 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[3]),
                                  self.act)
        # Bottleneck Block
        self.bottleneck_block = nn.Sequential(nn.Conv2d(self.c_list[3], self.c_list[3], kernel_size=3, stride=1, padding=1,
                                              groups=self.c_list[3]),
                                              self.act)
        self.stage4_de = SCSSD(in_channel_list=[self.c_list[3],self.c_list[3]],
                                     out_channel_list=[self.c_list[3],self.c_list[3]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode='decoder'
                                     )
        self.up_4 = nn.ConvTranspose2d(self.c_list[2], self.c_list[2], kernel_size=2, stride=2,output_padding=1)
        # self.up_4 = Upsample(2,self.c_list[2])
        self.gs_5 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[2]),
                                  self.act)
        # print("stage3_de,{0}   {1}".format(self.c_list[3],self.c_list[2]))
        self.stage3_de = SCSSD(in_channel_list=[self.c_list[2], self.c_list[2]],
                                     out_channel_list=[self.c_list[2], self.c_list[2]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode='decoder'
                                     )  # 2层
        # self.stage3_de = SCSSD_Layer(in_channel_list=[128, 96],
        #                              out_channel_list=[96, 64]) 
        self.up_3 = Upsample(2,self.c_list[1])
        self.gs_6 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[1]),
                                  self.act)

        self.stage2_de = SCSSD(in_channel_list=[self.c_list[1],self.c_list[1]],
                                     out_channel_list=[self.c_list[1],self.c_list[1]],
                                     ssd_config=self.ssd_config,
                                     split_nums=self.split_nums,
                                     mode='decoder'
                                     )  
        self.up_2 = Upsample(2,self.c_list[0])
        self.gs_7 = nn.Sequential(nn.GroupNorm(self.split_nums, self.c_list[0]),
                                  self.act)
        self.reconstruct = nn.Conv2d(self.c_list[0], self.out_channels, 3, 1, 1)

        self.up_brand = Upsampler(self.rs_factor, self.out_channels,self.act)
        self.reg = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        self.cls_brand = IntensityClassifier(act=self.act)
        # self.apply(self._init_weights)

        self.finally_act = nn.Softplus()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x,geo_lsm):

        # print(base)
        if self.is_cls:
            if not self.atten_config['MSCB'] and not self.atten_config['CBAM']:
                out1 = self.stage1_en(x)

                if self.dic_geo_lsm['geo_lsm']:
                    out_geo_lsm = self.geo_lsm(geo_lsm)
                    concate_data = torch.cat((out1,out_geo_lsm[-1]),dim=1)
                    out1 = self.contacate(concate_data)

                # print(f'out1 shape is: {out1.shape}')
                out2 = self.stage2_en(out1)
                out2 = self.max_pool_2(out2)
                out2 = self.gs_2(out2)

                out3 = self.stage3_en(out2)
                out3 = self.max_pool_3(out3)
                out3 = self.gs_3(out3)

                out4 = self.stage4_en(out3)
                out4 = self.max_pool_4(out4)
                out4 = self.gs_4(out4)


                out5 = self.stage4_de(out4)
                out5 = self.up_4(out5)
                out5 = self.gs_5(out5)

                out6 = self.stage3_de(out5)
                out6 = self.up_3(out6)
                out6 = self.gs_6(out6)

                out7 = self.stage2_de(out6)
                out7 = self.up_2(out7)
                out7 = self.gs_7(out7)

                out8 = self.reconstruct(out7)

                #cls
                cls_result = self.cls_brand(out8)
                # up
                up_result = self.up_brand(out8)
                reg_result = self.reg(up_result)
                reg_result = self.finally_act(reg_result)
            else:
                out1 = self.stage1_en(x)

                if self.dic_geo_lsm['geo_lsm']:
                    out_geo_lsm = self.geo_lsm(geo_lsm)
                    concate_data = torch.cat((out1,out_geo_lsm[-1]),dim=1)
                    out1 = self.contacate(concate_data)
                # print(f'out1 shape is: {out1.shape}')
                wavelet = self.wavelet_brand(x[:,0,:,:].unsqueeze(1))
                attn_1 =  self.att_1(out1)
                # print(out1.shape)
                out2 = self.stage2_en(out1)
                out2 = self.max_pool_2(out2)
                out2 = self.gs_2(out2)
                attn_2 = self.att_2(out2)

                out3 = self.stage3_en(out2)
                out3 = self.max_pool_3(out3)
                out3 = self.gs_3(out3)
                attn_3 = self.att_3(out3)

                out4 = self.stage4_en(out3)
                out4 = self.max_pool_4(out4)
                out4 = self.gs_4(out4)

                attn_4 = self.att_4(out4)

                out5 = self.stage4_de(torch.add(out4, attn_4))
                out5 = self.up_4(out5)

                out5 = self.gs_5(out5)

                out6 = self.stage3_de(torch.add(out5 , attn_3))
                out6 = self.up_3(out6)
                out6 = self.gs_6(out6)

                out7 = self.stage2_de(torch.add(out6 , attn_2))
                out7 = self.up_2(out7)
                out7 = self.gs_7(out7)

                out8 = self.reconstruct(torch.add(torch.add(out7 , attn_1),wavelet))

                #cls
                cls_result = self.cls_brand(out8)
                # up
                up_result = self.up_brand(out8)
                reg_result = self.reg(up_result)
                reg_result = self.finally_act(reg_result)

            if self.res:
                base = F.interpolate(x[:, 0, :, :].unsqueeze(1), scale_factor=self.rs_factor, mode='bicubic')
                reg = torch.add(reg_result,base) / 2
            # if self.is_cls:
            #     cls = self.cls(out8)
                return reg, cls_result
            else:
                return reg_result, cls_result
        else:
            if not self.atten_config['MSCB'] and not self.atten_config['CBAM']:
                out1 = self.stage1_en(x)

                if self.dic_geo_lsm['geo_lsm']:
                    out_geo_lsm = self.geo_lsm(geo_lsm)
                    concate_data = torch.cat((out1, out_geo_lsm[-1]), dim=1)
                    out1 = self.contacate(concate_data)
                # print(f'out1 shape is: {out1.shape}')
                out2 = self.stage2_en(out1)
                out2 = self.max_pool_2(out2)
                out2 = self.gs_2(out2)

                out3 = self.stage3_en(out2)
                out3 = self.max_pool_3(out3)
                out3 = self.gs_3(out3)

                out4 = self.stage4_en(out3)
                out4 = self.max_pool_4(out4)
                out4 = self.gs_4(out4)

                out5 = self.stage4_de(out4)
                out5 = self.up_4(out5)
                out5 = self.gs_5(out5)

                out6 = self.stage3_de(out5)
                out6 = self.up_3(out6)
                out6 = self.gs_6(out6)

                out7 = self.stage2_de(out6)
                out7 = self.up_2(out7)
                out7 = self.gs_7(out7)

                out8 = self.reconstruct(out7)

                # up
                up_result = self.up_brand(out8)
                reg_result = self.reg(up_result)
                reg_result = self.finally_act(reg_result)
            else:
                out1 = self.stage1_en(x)

                if self.dic_geo_lsm['geo_lsm']:
                    out_geo_lsm = self.geo_lsm(geo_lsm)
                    concate_data = torch.cat((out1, out_geo_lsm[-1]), dim=1)
                    out1 = self.contacate(concate_data)
                # print(f'out1 shape is: {out1.shape}')
                wavelet = self.wavelet_brand(x[:,0,:,:].unsqueeze(1))
                attn_1 = self.att_1(out1)
                # print(out1.shape)
                out2 = self.stage2_en(out1)
                out2 = self.max_pool_2(out2)
                out2 = self.gs_2(out2)
                attn_2 = self.att_2(out2)

                out3 = self.stage3_en(out2)
                out3 = self.max_pool_3(out3)
                out3 = self.gs_3(out3)
                attn_3 = self.att_3(out3)

                out4 = self.stage4_en(out3)
                out4 = self.max_pool_4(out4)
                out4 = self.gs_4(out4)

                attn_4 = self.att_4(out4)

                # outb = self.bottleneck_block(out4)

                # out5 = self.stage4_de(torch.add(outb , attn_4))

                out5 = self.stage4_de(torch.add(out4, attn_4))
                out5 = self.up_4(out5)
                # print(f'out5 shape is: {out5.shape}')
                out5 = self.gs_5(out5)
                # print(f'out5 shape is: {out5.shape}')
                # print(f'attn_3 shape is: {attn_3.shape}')
                out6 = self.stage3_de(torch.add(out5, attn_3))
                out6 = self.up_3(out6)
                out6 = self.gs_6(out6)

                out7 = self.stage2_de(torch.add(out6, attn_2))
                out7 = self.up_2(out7)
                out7 = self.gs_7(out7)

                out8 = self.reconstruct(torch.add(torch.add(out7, attn_1), wavelet))

                # up
                up_result = self.up_brand(out8)
                reg_result = self.reg(up_result)

                reg_result = self.finally_act(reg_result)
            if self.res:
                base = F.interpolate(x[:, 0, :, :].unsqueeze(1), scale_factor=self.rs_factor, mode='bicubic')
                reg = torch.add(reg_result, base) / 2
                # if self.is_cls:
                #     cls = self.cls(out8)
                return reg
            else:
                return reg_result
