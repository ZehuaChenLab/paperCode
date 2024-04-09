"""
xz & 2023/12/8 18:41
"""
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from networks.DSConv import DSConv
from functools import partial
from mmcv.cnn import build_norm_layer
from timm.models.layers import trunc_normal_, DropPath

nonlinearity = partial(F.relu, inplace=True)

class StripUnet(nn.Module):  # strip-unet
    def __init__(self, num_classes=1):
        super(StripUnet, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        # 1024*3
        self.firstconv = resnet.conv1  # 512*64
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # 256*64

        #self.ripe = RIPE(64,64)

        self.encoder1 = resnet.layer1  # 256*64
        self.encoder2 = resnet.layer2  # 128*128
        self.encoder3 = resnet.layer3  # 64*256
        self.encoder4 = resnet.layer4  # 32*512

        self.msff = MSFF(512)

        self.dam1 = DAM(256, 64)   # 256*64
        self.dam2 = DAM(128, 128)   # 128*128
        self.dam3 = DAM(64, 256)    # 64*256
        self.dam4 = DAM(32, 512)   # 32*512

        self.convsam0 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.convsam1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.convsam2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.convsam3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)

        self.bnsam0 = nn.BatchNorm2d(64)
        self.bnsam1 = nn.BatchNorm2d(128)
        self.bnsam2 = nn.BatchNorm2d(256)
        self.bnsam3 = nn.BatchNorm2d(512)

        self.jiangwei111 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.jiangwei211 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.jiangwei311 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.jiangwei411 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)

        self.sfem4 = sfem(512, d=1)  # 32*512 变大降维后64*256
        self.sfem3 = sfem(256, d=2)  # 64*256 变大降维后128*128
        self.sfem2 = sfem(128, d=4)  # 128*128 变大降维后256*64
        self.sfem1 = sfem(64 , d=8) # 256*64 变大后512*64 不用降维

        self.dconv4 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1)
        self.dconv3 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        self.dconv2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.dconv1 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)

        self.decoder4 = DDSCDecoderBlock(filters[3], filters[2])  # 64*256
        self.decoder3 = DDSCDecoderBlock(filters[2], filters[1])  # 128*128
        self.decoder2 = DDSCDecoderBlock(filters[1], filters[0])  # 256*64
        self.decoder1 = DDSCDecoderBlock(filters[0], filters[0])  # 512*64

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 1024*32
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)  # 1024*32
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)  # 1024*1

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)  # 512*64
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x) # 256*64

        #ripe = self.ripe(x1)

        e1 = self.encoder1(x)   # 256*64
        e2 = self.encoder2(e1)   # 128*128
        e3 = self.encoder3(e2)   # 64*256
        e4 = self.encoder4(e3)   # 32*512

        # Attention
        #e00 = self.convsam0(e0)  # e00 256*64
        #e00 = nonlinearity(self.bnsam0(e0))
        sam1 = self.dam1(e1,x)   # 256*64

        sam11 = self.convsam1(sam1)  # sam11 128*128
        sam11 = nonlinearity(self.bnsam1(sam11))
        sam2 = self.dam2(e2,sam11)   # 128*128

        sam22 = self.convsam2(sam2)   #sam22 64*256
        sam22 = nonlinearity(self.bnsam2(sam22))
        sam3 = self.dam3(e3,sam22)   # 64*256

        sam33 = self.convsam3(sam3)    #sam33 32*512
        sam33 = nonlinearity(self.bnsam3(sam33))

        sam4 = self.dam4(e4,sam33)   # 32*512

        # Center
        e4 = self.msff(e4 + sam4)  # 32*512

        #sfem
        c4 = self.sfem4(sam4+e4)   #c4 32*512
        c44 = self.jiangwei111(c4)
        c44 = self.dconv4(c44)   #c44 64*256

        c3 = self.sfem3(sam3+c44)   #c3 64*256
        c33 = self.jiangwei211(c3)
        c33 = self.dconv3(c33)   #c33 128*128

        c2 = self.sfem2(sam2+c33)   #c2 128*128
        c22 = self.jiangwei311(c2)
        c22 = self.dconv2(c22)   #c22 256*64

        c1 = self.sfem1(sam1+c22)   #c1 256*64

        d4 = e4 + c4
        # Decoder
        d3 = self.decoder4(d4) + c3
        d2 = self.decoder3(d3) + c2
        d1 = self.decoder2(d2) + c1
        d0 = self.decoder1(d1)

        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)



class RIPE(nn.Module):   #HWC->H/2 W/2 C
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)

        self.bn = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(3*in_channels, out_channels, kernel_size=1)

        self.strpool = StripPooling(in_channels=64, pool_size=(20, 12),
                                    norm_layer=nn.BatchNorm2d,
                                    up_kwargs={'mode': 'bilinear', 'align_corners': True})

    def forward(self, x):
        x1 = self.conv3(x)
        x2 = self.conv5(x)
        x3 = self.conv7(x)

        x1 = self.bn(x1)
        x2 = self.bn(x2)
        x3 = self.bn(x3)

        x1 = nonlinearity(x1)
        x2 = nonlinearity(x2)
        x3 = nonlinearity(x3)

        x1 = self.strpool(x1)
        x2 = self.strpool(x2)
        x3 = self.strpool(x3)
        x = torch.cat((x1, x2, x3), 1)

        x = self.conv1(x)
        x = nonlinearity(self.bn(x))

        return x

class StripPooling(nn.Module): #HWC不变
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels / 4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels * 2, in_channels, 1, bias=False),
                                   norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)


class DAM(nn.Module):
    def __init__(self, c_size, channel):
        super(DAM, self).__init__()
        self.GPA1 = nn.AdaptiveAvgPool2d((c_size, c_size))
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.sigmoid1 = nn.Sigmoid()

        self.GPA2 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv4 = nn.Conv2d(in_channels=channel, out_channels=channel * 2, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(channel * 2)
        self.conv5 = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1)
        self.stripatt = TDAtt(channel)

    def forward(self, c_x, d_x):
        x1 = self.GPA1(c_x)
        x1 = self.conv1(x1)
        x1 = nonlinearity(self.bn1(x1))
        x1 = self.sigmoid1(x1)
        x1 = torch.mul(c_x, x1)
        x2 = self.GPA2(c_x)
        x2 = self.conv2(x2)
        x2 = nonlinearity(x2)
        x2 = self.conv3(x2)
        x2 = self.sigmoid2(x2)
        x2 = torch.mul(c_x, x2)
        x2 = c_x + x1 + x2
        d_x = self.stripatt(d_x)
        x = torch.cat((x2, d_x), dim=1)
        x = self.conv5(x)
        x = nonlinearity(self.bn2(x))

        return x


class TDAtt(nn.Module):  # 条带注意力 很猛
    def __init__(self, dim):
        super(TDAtt, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 15), padding=(0, 7), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (15, 1), padding=(7, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * u


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class MSFF(nn.Module):
    def __init__(self, channel):
        super(MSFF, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

        self.block1 = Block(channel, mlp_ratio=4, drop_path=0.2)
        self.block2 = Block(channel, mlp_ratio=4, drop_path=0.2)
        self.block3 = Block(channel, mlp_ratio=4, drop_path=0.2)
        self.block4 = Block(channel, mlp_ratio=4, drop_path=0.2)
        self.block5 = Block(channel, mlp_ratio=4, drop_path=0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))

        dilate1_out = self.block1(dilate1_out)
        dilate2_out = self.block2(dilate2_out)
        dilate3_out = self.block3(dilate3_out)
        dilate4_out = self.block4(dilate4_out)
        x = self.block5(x)

        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.1):
        super().__init__()

        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class sfem(nn.Module):
    def __init__(self, nIn, d=1):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, 3, 1, padding=1, bn_acti=True)

        self.dconv3x1_1_1 = Conv(nIn // 4, nIn // 16, (3, 1), 1,
                                 padding=(1, 0), groups=nIn // 16, bn_acti=True)
        self.dconv1x3_1_1 = Conv(nIn // 16, nIn // 16, (1, 3), 1,
                                 padding=(0, 1), groups=nIn // 16, bn_acti=True)

        self.dconv5x1_1_2 = Conv(nIn // 16, nIn // 16, (5, 1), 1,
                                 padding=(2, 0), groups=nIn // 16, bn_acti=True)
        self.dconv1x5_1_2 = Conv(nIn // 16, nIn // 16, (1, 5), 1,
                                 padding=(0, 2), groups=nIn // 16, bn_acti=True)

        self.dconv7x1_1_3 = Conv(nIn // 16, nIn // 8, (7, 1), 1,
                                 padding=(3, 0), groups=nIn // 16, bn_acti=True)
        self.dconv1x7_1_3 = Conv(nIn // 8, nIn // 8, (1, 7), 1,
                                 padding=(0, 3), groups=nIn // 8, bn_acti=True)

        self.dconv3x1_2_1 = Conv(nIn // 4, nIn // 16, (3, 1), 1,
                                 padding=(int(d / 4 + 1), 0), dilation=(int(d / 4 + 1), 1), groups=nIn // 16,
                                 bn_acti=True)
        self.dconv1x3_2_1 = Conv(nIn // 16, nIn // 16, (1, 3), 1,
                                 padding=(0, int(d / 4 + 1)), dilation=(1, int(d / 4 + 1)), groups=nIn // 16,
                                 bn_acti=True)

        self.dconv5x1_2_2 = Conv(nIn // 16, nIn // 16, (5, 1), 1,
                                 padding=(2*int(d / 4 + 1), 0), dilation=(int(d / 4 + 1), 1), groups=nIn // 16,
                                 bn_acti=True)
        self.dconv1x5_2_2 = Conv(nIn // 16, nIn // 16, (1, 5), 1,
                                 padding=(0, 2*int(d / 4 + 1)), dilation=(1, int(d / 4 + 1)), groups=nIn // 16,
                                 bn_acti=True)

        self.dconv7x1_2_3 = Conv(nIn // 16, nIn // 8, (7, 1), 1,
                                 padding=(3*int(d / 4 + 1), 0), dilation=(int(d / 4 + 1), 1), groups=nIn // 16,
                                 bn_acti=True)
        self.dconv1x7_2_3 = Conv(nIn // 8, nIn // 8, (1, 7), 1,
                                 padding=(0, 3*int(d / 4 + 1 )), dilation=(1, int(d / 4 + 1)), groups=nIn // 8,
                                 bn_acti=True)

        self.dconv3x1_3_1 = Conv(nIn // 4, nIn // 16, (3, 1), 1,
                                 padding=(int(d / 2 + 1), 0), dilation=(int(d / 2 + 1), 1), groups=nIn // 16,
                                 bn_acti=True)
        self.dconv1x3_3_1 = Conv(nIn // 16, nIn // 16, (1, 3), 1,
                                 padding=(0, int(d / 2 + 1)), dilation=(1, int(d / 2 + 1)), groups=nIn // 16,
                                 bn_acti=True)

        self.dconv5x1_3_2 = Conv(nIn // 16, nIn // 16, (5, 1), 1,
                                 padding=(2*int(d / 2 + 1), 0), dilation=(int(d / 2 + 1), 1), groups=nIn // 16,
                                 bn_acti=True)
        self.dconv1x5_3_2 = Conv(nIn // 16, nIn // 16, (1, 5), 1,
                                 padding=(0, 2*int(d / 2 + 1)), dilation=(1, int(d / 2 + 1)), groups=nIn // 16,
                                 bn_acti=True)

        self.dconv7x1_3_3 = Conv(nIn // 16, nIn // 8, (7, 1), 1,
                                 padding=(3*int(d / 2 + 1), 0), dilation=(int(d / 2 + 1), 1), groups=nIn // 16,
                                 bn_acti=True)
        self.dconv1x7_3_3 = Conv(nIn // 8, nIn // 8, (1, 7), 1,
                                 padding=(0, 3*int(d / 2 + 1)), dilation=(1, int(d / 2 + 1)), groups=nIn // 8,
                                 bn_acti=True)

        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)
        self.bnlast = nn.BatchNorm2d(nIn)

    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)

        o1_1 = self.dconv3x1_1_1(inp)
        o1_1 = self.dconv1x3_1_1(o1_1)
        o1_2 = self.dconv5x1_1_2(o1_1)
        o1_2 = self.dconv1x5_1_2(o1_2)
        o1_3 = self.dconv7x1_1_3(o1_2)
        o1_3 = self.dconv1x7_1_3(o1_3)

        o2_1 = self.dconv3x1_2_1(inp)
        o2_1 = self.dconv1x3_2_1(o2_1)
        o2_2 = self.dconv5x1_2_2(o2_1)
        o2_2 = self.dconv1x5_2_2(o2_2)
        o2_3 = self.dconv7x1_2_3(o2_2)
        o2_3 = self.dconv1x7_2_3(o2_3)

        o3_1 = self.dconv3x1_3_1(inp)
        o3_1 = self.dconv1x3_3_1(o3_1)
        o3_2 = self.dconv5x1_3_2(o3_1)
        o3_2 = self.dconv1x5_3_2(o3_2)
        o3_3 = self.dconv7x1_3_3(o3_2)
        o3_3 = self.dconv1x7_3_3(o3_3)

        output_1 = torch.cat([o1_1, o1_2, o1_3], 1)
        output_2 = torch.cat([o2_1, o2_2, o2_3], 1)
        output_3 = torch.cat([o3_1, o3_2, o3_3], 1)

        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        output = torch.cat([inp, ad1, ad2, ad3], 1)
        output = self.conv1x1(output)
        output = self.bn_relu_2(output)

        return output + input


class DDSCDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DDSCDecoderBlock, self).__init__()
        self.deconv2 = nn.ConvTranspose2d(in_ch, in_ch // 2, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_ch // 2)
        self.block1 = nn.Sequential(
            DSConv(in_ch // 2, in_ch // 2, kernel_size=3, extend_scope=3, morph=0, if_offset=True, device="cuda"),
            nn.BatchNorm2d(in_ch // 2))
        self.block2 = nn.Sequential(
            DSConv(in_ch // 2, in_ch // 2, kernel_size=5, extend_scope=5, morph=1, if_offset=True, device="cuda"),
            nn.BatchNorm2d(in_ch // 2))
        self.block3 = nn.Sequential(nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=7, padding=3, bias=False),
                                    nn.BatchNorm2d(in_ch // 2))
        self.conv3 = nn.Conv2d( 3*in_ch//2, out_ch, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.deconv2(x)
        x = nonlinearity(self.norm2(x))
        x1 = nonlinearity(self.block1(x))
        x2 = nonlinearity(self.block2(x))
        x3 = nonlinearity(self.block3(x))
        x = torch.cat((x1, x2, x3), 1)
        x = self.conv3(x)
        x = nonlinearity(self.norm3(x))
        return x