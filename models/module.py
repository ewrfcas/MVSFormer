import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")


class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


autocast = torch.cuda.amp.autocast if torch.__version__ >= '1.6.0' else identity_with


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, norm_type='IN', **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        if norm_type == 'IN':
            self.bn = nn.InstanceNorm2d(out_channels, momentum=bn_momentum) if bn else None
        elif norm_type == 'BN':
            self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu:
            y = F.leaky_relu(y, 0.1, inplace=True)
        return y

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class ConvBnReLU(nn.Module):
    """Implements 2d Convolution + batch normalization + ReLU"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            pad: int = 1,
            dilation: int = 1,
    ) -> None:
        """initialization method for convolution2D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class FPNEncoder(nn.Module):
    def __init__(self, feat_chs, norm_type='BN'):
        super(FPNEncoder, self).__init__()
        self.conv00 = Conv2d(3, feat_chs[0], 7, 1, padding=3, norm_type=norm_type)
        self.conv01 = Conv2d(feat_chs[0], feat_chs[0], 5, 1, padding=2, norm_type=norm_type)

        self.downsample1 = Conv2d(feat_chs[0], feat_chs[1], 5, stride=2, padding=2, norm_type=norm_type)
        self.conv10 = Conv2d(feat_chs[1], feat_chs[1], 3, 1, padding=1, norm_type=norm_type)
        self.conv11 = Conv2d(feat_chs[1], feat_chs[1], 3, 1, padding=1, norm_type=norm_type)

        self.downsample2 = Conv2d(feat_chs[1], feat_chs[2], 5, stride=2, padding=2, norm_type=norm_type)
        self.conv20 = Conv2d(feat_chs[2], feat_chs[2], 3, 1, padding=1, norm_type=norm_type)
        self.conv21 = Conv2d(feat_chs[2], feat_chs[2], 3, 1, padding=1, norm_type=norm_type)

        self.downsample3 = Conv2d(feat_chs[2], feat_chs[3], 3, stride=2, padding=1, norm_type=norm_type)
        self.conv30 = Conv2d(feat_chs[3], feat_chs[3], 3, 1, padding=1, norm_type=norm_type)
        self.conv31 = Conv2d(feat_chs[3], feat_chs[3], 3, 1, padding=1, norm_type=norm_type)

    def forward(self, x):
        conv00 = self.conv00(x)
        conv01 = self.conv01(conv00)
        down_conv0 = self.downsample1(conv01)
        conv10 = self.conv10(down_conv0)
        conv11 = self.conv11(conv10)
        down_conv1 = self.downsample2(conv11)
        conv20 = self.conv20(down_conv1)
        conv21 = self.conv21(conv20)
        down_conv2 = self.downsample3(conv21)
        conv30 = self.conv30(down_conv2)
        conv31 = self.conv31(conv30)

        return [conv01, conv11, conv21, conv31]


class FPNDecoder(nn.Module):
    def __init__(self, feat_chs):
        super(FPNDecoder, self).__init__()
        final_ch = feat_chs[-1]
        self.out0 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[3], kernel_size=1), nn.BatchNorm2d(feat_chs[3]), Swish())

        self.inner1 = nn.Conv2d(feat_chs[2], final_ch, 1)
        self.out1 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[2], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[2]), Swish())

        self.inner2 = nn.Conv2d(feat_chs[1], final_ch, 1)
        self.out2 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[1], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[1]), Swish())

        self.inner3 = nn.Conv2d(feat_chs[0], final_ch, 1)
        self.out3 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[0], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[0]), Swish())

    def forward(self, conv01, conv11, conv21, conv31):
        intra_feat = conv31
        out0 = self.out0(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv21)
        out1 = self.out1(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(conv11)
        out2 = self.out2(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(conv01)
        out3 = self.out3(intra_feat)

        return [out0, out1, out2, out3]


class FPNDecoderV2(nn.Module):
    def __init__(self, feat_chs):
        super(FPNDecoderV2, self).__init__()
        self.out1 = nn.Sequential(nn.Conv2d(feat_chs[3] * 2, feat_chs[3], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[3]), Swish())
        self.upsample1 = nn.Sequential(nn.ConvTranspose2d(feat_chs[3], feat_chs[2], kernel_size=4, stride=2, padding=1),
                                       nn.BatchNorm2d(feat_chs[2]), nn.ReLU(True))

        self.out2 = nn.Sequential(nn.Conv2d(feat_chs[2] * 2, feat_chs[2], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[2]), Swish())
        self.upsample2 = nn.Sequential(nn.ConvTranspose2d(feat_chs[2], feat_chs[1], kernel_size=4, stride=2, padding=1),
                                       nn.BatchNorm2d(feat_chs[1]), nn.ReLU(True))

        self.out3 = nn.Sequential(nn.Conv2d(feat_chs[1] * 2, feat_chs[1], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[1]), Swish())
        self.upsample3 = nn.Sequential(nn.ConvTranspose2d(feat_chs[1], feat_chs[0], kernel_size=4, stride=2, padding=1),
                                       nn.BatchNorm2d(feat_chs[0]), nn.ReLU(True))

        self.out4 = nn.Sequential(nn.Conv2d(feat_chs[0], feat_chs[0], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[0]), Swish())

    def forward(self, conv01, conv11, conv21, conv31, vit1, vit2, vit3):
        out1 = self.out1(torch.cat([conv31, vit1], dim=1))  # [B,64,H/8,W/8]

        out2 = self.upsample1(out1) + conv21
        out2 = self.out2(torch.cat([out2, vit2], dim=1))  # [B,32,H/4,W/4]

        out3 = self.upsample2(out2) + conv11
        out3 = self.out3(torch.cat([out3, vit3], dim=1))  # [B,16,H/2,W/2]

        out4 = self.upsample3(out3) + conv01
        out4 = self.out4(out4)  # [B,8,H,W]

        return [out1, out2, out3, out4]


class VITDecoderStage4(nn.Module):
    def __init__(self, args):
        super(VITDecoderStage4, self).__init__()
        ch, vit_ch = args['out_ch'], args['vit_ch']
        self.multi_scale_decoder = args.get('multi_scale_decoder', False)
        assert args['att_fusion'] is True
        self.attn = AttentionFusionSimple(vit_ch, ch * 4, args['nhead'])
        if self.multi_scale_decoder:
            self.decoder1 = nn.Sequential(nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
                                          nn.BatchNorm2d(ch * 2), nn.GELU(),
                                          nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1))

            self.decoder2 = nn.Sequential(nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
                                          nn.BatchNorm2d(ch * 2), nn.GELU(),
                                          nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
                                          nn.BatchNorm2d(ch), nn.GELU(),
                                          nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1))

            self.decoder3 = nn.Sequential(nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
                                          nn.BatchNorm2d(ch * 2), nn.GELU(),
                                          nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
                                          nn.BatchNorm2d(ch), nn.GELU(),
                                          nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1),
                                          nn.BatchNorm2d(ch // 2), nn.GELU(),
                                          nn.ConvTranspose2d(ch // 2, ch // 4, 4, stride=2, padding=1))
        else:
            self.decoder1 = nn.Sequential(nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
                                          nn.BatchNorm2d(ch * 2), nn.GELU(),
                                          nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1))
            self.decoder2 = nn.Sequential(nn.BatchNorm2d(ch), nn.GELU(),
                                          nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1))
            self.decoder3 = nn.Sequential(nn.BatchNorm2d(ch // 2), nn.GELU(),
                                          nn.ConvTranspose2d(ch // 2, ch // 4, 4, stride=2, padding=1))

    def forward(self, x, att):
        x = self.attn(x, att)
        if self.multi_scale_decoder:
            out1 = self.decoder1(x)
            out2 = self.decoder2(x)
            out3 = self.decoder3(x)
        else:
            out1 = self.decoder1(x)
            out2 = self.decoder2(out1)
            out3 = self.decoder3(out2)

        return out1, out2, out3


class VITDecoderStage4Single(nn.Module):
    def __init__(self, args):
        super(VITDecoderStage4Single, self).__init__()
        ch, vit_ch = args['out_ch'], args['vit_ch']
        assert args['att_fusion'] is True
        self.attn = AttentionFusionSimple(vit_ch, ch * 4, args['nhead'])
        self.decoder = nn.Sequential(nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
                                     nn.BatchNorm2d(ch * 2), nn.GELU(),
                                     nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
                                     nn.BatchNorm2d(ch), nn.GELU())

    def forward(self, x, att):
        x = self.attn(x, att)
        x = self.decoder(x)

        return x


class VITDecoderStage4NoAtt(nn.Module):
    def __init__(self, args):
        super(VITDecoderStage4NoAtt, self).__init__()
        ch, vit_ch = args['out_ch'], args['vit_ch']
        self.down_sample = nn.Sequential(nn.Conv2d(vit_ch, ch * 4, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(ch * 4), nn.GELU())
        self.decoder = nn.Sequential(nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
                                     nn.BatchNorm2d(ch * 2), nn.GELU(),
                                     nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
                                     nn.BatchNorm2d(ch), nn.GELU())

    def forward(self, x, att=None):
        x = self.down_sample(x)
        x = self.decoder(x)

        return x


class TwinDecoderStage4(nn.Module):
    def __init__(self, args):
        super(TwinDecoderStage4, self).__init__()
        ch, vit_chs = args['out_ch'], args['vit_ch']
        ch = ch * 4  # 256
        self.upsampler0 = nn.Sequential(nn.ConvTranspose2d(vit_chs[-1], ch, 4, stride=2, padding=1),
                                        nn.BatchNorm2d(ch), nn.GELU())  # 256
        self.inner1 = nn.Conv2d(vit_chs[-2], ch, kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Sequential(nn.Conv2d(ch, ch // 2, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(ch // 2), nn.ReLU(True))  # 256->128

        self.inner2 = nn.Conv2d(vit_chs[-3], ch // 2, kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Sequential(nn.Conv2d(ch // 2, ch // 4, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(ch // 4), nn.ReLU(True))  # 128->64

        self.inner3 = nn.Conv2d(vit_chs[-4], ch // 4, kernel_size=1, stride=1, padding=0)
        self.smooth3 = nn.Sequential(nn.Conv2d(ch // 4, ch // 4, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(ch // 4), Swish())  # 64->64

    def forward(self, x1, x2, x3, x4):  # in:[1/8 ~ 1/64] out:[1/2,1/4,1/8]
        x = self.smooth1(self.upsampler0(x4) + self.inner1(x3))  # 1/64->1/32
        x = self.smooth2(F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False) + self.inner2(x2))  # 1/32->1/16
        x = self.smooth3(F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False) + self.inner3(x1))  # 1/16->1/8

        return x


class TwinDecoderStage4V2(nn.Module):
    def __init__(self, args):
        super(TwinDecoderStage4V2, self).__init__()
        ch, vit_chs = args['out_ch'], args['vit_ch']
        ch = ch * 4  # 256
        self.upsampler0 = nn.Sequential(nn.ConvTranspose2d(vit_chs[-1], ch, 4, stride=2, padding=1),
                                        nn.BatchNorm2d(ch), nn.GELU())  # 256
        self.inner1 = nn.Conv2d(vit_chs[-2], ch, kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Sequential(nn.Conv2d(ch, ch // 2, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(ch // 2), nn.GELU())  # 256->128

        self.inner2 = nn.Conv2d(vit_chs[-3], ch // 2, kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Sequential(nn.Conv2d(ch // 2, ch // 4, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(ch // 4), nn.GELU())  # 128->64

        self.inner3 = nn.Conv2d(vit_chs[-4], ch // 4, kernel_size=1, stride=1, padding=0)
        self.smooth3 = nn.Sequential(nn.Conv2d(ch // 4, ch // 4, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(ch // 4), nn.GELU())  # 64->64

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(ch // 4, ch // 8, 4, stride=2, padding=1),
                                      nn.BatchNorm2d(ch // 8), nn.GELU())
        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(ch // 8, ch // 16, 4, stride=2, padding=1),
                                      nn.BatchNorm2d(ch // 16), nn.GELU())

    def forward(self, x1, x2, x3, x4):  # in:[1/8 ~ 1/64] out:[1/2,1/4,1/8]
        x = self.smooth1(self.upsampler0(x4) + self.inner1(x3))  # 1/64->1/32
        x = self.smooth2(F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False) + self.inner2(x2))  # 1/32->1/16
        out1 = self.smooth3(F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False) + self.inner3(x1))  # 1/16->1/8
        out2 = self.decoder1(out1)
        out3 = self.decoder2(out2)

        return out1, out2, out3


class AttentionFusionSimple(nn.Module):
    def __init__(self, vit_ch, out_ch, nhead):
        super(AttentionFusionSimple, self).__init__()
        self.conv_l = nn.Sequential(nn.Conv2d(vit_ch + nhead, vit_ch, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(vit_ch))
        self.conv_r = nn.Sequential(nn.Conv2d(vit_ch, vit_ch, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(vit_ch))
        self.act = Swish()
        self.proj = nn.Conv2d(vit_ch, out_ch, kernel_size=1)

    def forward(self, x, att):
        # x:[B,C,H,W]; att:[B,nh,H,W]
        x1 = self.act(self.conv_l(torch.cat([x, att], dim=1)))
        att = torch.mean(att, dim=1, keepdim=True)
        x2 = self.act(self.conv_r(x * att))
        x = self.proj(x1 * x2)
        return x


class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels, last_layer=True):
        super(CostRegNet, self).__init__()
        self.last_layer = last_layer

        self.conv1 = Conv3d(in_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)
        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)
        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        if in_channels != base_channels:
            self.inner = nn.Conv3d(in_channels, base_channels, 1, 1)
        else:
            self.inner = nn.Identity()

        if self.last_layer:
            self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = self.inner(conv0) + self.conv11(x)
        if self.last_layer:
            x = self.prob(x)
        return x


class CostRegNet2D(nn.Module):
    def __init__(self, in_channels, base_channel=8):
        super(CostRegNet2D, self).__init__()
        self.conv1 = Conv3d(in_channels, base_channel * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv2 = Conv3d(base_channel * 2, base_channel * 2, padding=1)

        self.conv3 = Conv3d(base_channel * 2, base_channel * 4, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv4 = Conv3d(base_channel * 4, base_channel * 4, padding=1)

        self.conv5 = Conv3d(base_channel * 4, base_channel * 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv6 = Conv3d(base_channel * 8, base_channel * 8, padding=1)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 8, base_channel * 4, kernel_size=(1, 3, 3), padding=(0, 1, 1), output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 4, base_channel * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1), output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 2, base_channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(base_channel, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x


class CostRegNet3D(nn.Module):
    def __init__(self, in_channels, base_channel=8):
        super(CostRegNet3D, self).__init__()
        self.conv1 = Conv3d(in_channels, base_channel * 2, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv2 = Conv3d(base_channel * 2, base_channel * 2, padding=1)

        self.conv3 = Conv3d(base_channel * 2, base_channel * 4, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv4 = Conv3d(base_channel * 4, base_channel * 4, padding=1)

        self.conv5 = Conv3d(base_channel * 4, base_channel * 8, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv6 = Conv3d(base_channel * 8, base_channel * 8, padding=1)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 8, base_channel * 4, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 4, base_channel * 2, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 2, base_channel, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        if in_channels != base_channel:
            self.inner = nn.Conv3d(in_channels, base_channel, 1, 1)
        else:
            self.inner = nn.Identity()

        self.prob = nn.Conv3d(base_channel, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = self.inner(conv0) + self.conv11(x)
        x = self.prob(x)

        return x


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth


def conf_regression(p, n=4):
    ndepths = p.size(1)
    with torch.no_grad():
        # photometric confidence
        if n % 2 == 1:
            prob_volume_sum4 = n * F.avg_pool3d(F.pad(p.unsqueeze(1), pad=[0, 0, 0, 0, n // 2, n // 2]),
                                                (n, 1, 1), stride=1, padding=0).squeeze(1)
        else:
            prob_volume_sum4 = n * F.avg_pool3d(F.pad(p.unsqueeze(1), pad=[0, 0, 0, 0, n // 2 - 1, n // 2]),
                                                (n, 1, 1), stride=1, padding=0).squeeze(1)
        depth_index = depth_regression(p.detach(), depth_values=torch.arange(ndepths, device=p.device, dtype=torch.float)).long()
        depth_index = depth_index.clamp(min=0, max=ndepths - 1)
        conf = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1))
    return conf.squeeze(1)


def init_range(cur_depth, ndepths, device, dtype, H, W):
    cur_depth_min = cur_depth[:, 0]  # (B,)
    cur_depth_max = cur_depth[:, -1]
    new_interval = (cur_depth_max - cur_depth_min) / (ndepths - 1)  # (B, )
    new_interval = new_interval[:, None, None]  # B H W
    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepths, device=device, dtype=dtype,
                                                                     requires_grad=False).reshape(1, -1) * new_interval.squeeze(1))  # (B, D)
    depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # (B, D, H, W)
    return depth_range_samples


def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1. / cur_depth[:, -1]
    itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepths - 1)  # 1 D H W
    inverse_depth_hypo = inverse_depth_max[:, None, None, None] + (inverse_depth_min - inverse_depth_max)[:, None, None, None] * itv

    return 1. / inverse_depth_hypo


def schedule_inverse_range(depth, depth_hypo, ndepths, split_itv, H, W):
    last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]
    inverse_min_depth = 1 / depth + split_itv * last_depth_itv  # B H W
    inverse_max_depth = 1 / depth - split_itv * last_depth_itv  # B H W
    # cur_depth_min, (B, H, W)
    # cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype,
                       requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H // 2, W // 2) / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:, None, :, :] + (inverse_min_depth - inverse_max_depth)[:, None, :, :] * itv  # B D H W
    inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return 1. / inverse_depth_hypo


def init_inverse_range_eth3d(cur_depth, ndepths, device, dtype, H, W):
    cur_depth = torch.clamp(cur_depth, min=0.01, max=50)

    inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1. / cur_depth[:, -1]

    itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepths - 1)  # 1 D H W
    inverse_depth_hypo = inverse_depth_max[:, None, None, None] + (inverse_depth_min - inverse_depth_max)[:, None, None, None] * itv

    return 1. / inverse_depth_hypo


def schedule_inverse_range_eth3d(depth, depth_hypo, ndepths, split_itv, H, W):
    last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]
    inverse_min_depth = 1 / depth + split_itv * last_depth_itv  # B H W
    inverse_max_depth = 1 / depth - split_itv * last_depth_itv  # B H W 只有他可能是负数！

    is_neg = (inverse_max_depth < 0.02).float()
    inverse_max_depth = inverse_max_depth - (inverse_max_depth - 0.02) * is_neg
    inverse_min_depth = inverse_min_depth - (inverse_max_depth - 0.02) * is_neg

    # cur_depth_min, (B, H, W)
    # cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype,
                       requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H // 2, W // 2) / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:, None, :, :] + (inverse_min_depth - inverse_max_depth)[:, None, :, :] * itv  # B D H W
    inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return 1. / inverse_depth_hypo


def schedule_range(cur_depth, ndepth, depth_inteval_pixel, H, W):
    # shape, (B, H, W)
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel[:, None, None])  # (B, H, W)
    cur_depth_min = torch.clamp_min(cur_depth_min, 0.01)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel[:, None, None])
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype,
                                                                     requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
    depth_range_samples = F.interpolate(depth_range_samples.unsqueeze(1), [ndepth, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return depth_range_samples
