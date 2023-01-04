# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

class ASPFModule(nn.ModuleList):
    """Atrous Spatial Pyramid Fusion (ASPF) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPFModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspf_outs = []
        for aspf_module in self:
            aspf_outs.append(aspf_module(x))

        return aspf_outs

    
from functools import reduce
class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()  
        for i in range(M):
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False), #groups=32
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(output_size = 1)
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False) 
        self.softmax=nn.Softmax(dim=1) 
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))    #[batch_size,out_channels,H,W]
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) #   [batch_size,channel,H,W]
        # print(U.size())            
        s=self.global_pool(U)     # [batch_size,channel,1,1]
        # print(s.size())
        z=self.fc1(s)  # S->Z   # [batch_size,d,1,1]
        # print(z.size())
        a_b=self.fc2(z) # Z->a，b  [batch_size,out_channels*M,1,1]
        # print(a_b.size())
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #[batch_size,M,out_channels,1]  
        # print(a_b.size())
        a_b=self.softmax(a_b) # softmax [batch_size,M,out_channels,1]  
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b    [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        # print(a_b[0].size())
        # print(a_b[1].size())
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        V=list(map(lambda x,y:x*y,output,a_b)) # [batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V=reduce(lambda x,y:x+y,V) #   [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        return V    # [batch_size,out_channels,H,W]
    
@HEADS.register_module()
class UPerHeadASPFSK(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,  dilations=(1, 6, 12, 18),**kwargs): #pool_scales=(1, 2, 3, 6),
        super(UPerHeadASPFSK, self).__init__(
            input_transform='multiple_select', **kwargs)

        self.aspf_modules = ASPFModule(
            dilations,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.bottleneck1 = ConvModule(
            len(dilations) * self.channels+768,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.convSK=SKConv(512,512)

    def aspf_forward(self, inputs):
        """Forward function of ASPF module."""
        x = inputs[-1]
        psp_outs = [x]  #
        psp_outs.extend(self.aspf_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck1(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.aspf_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals) #4
        for i in range(used_backbone_levels - 1, 0, -1): #range(3,0,-1)  i= 3,2,1
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        feats= self.convSK(feats)



        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)


        return output
