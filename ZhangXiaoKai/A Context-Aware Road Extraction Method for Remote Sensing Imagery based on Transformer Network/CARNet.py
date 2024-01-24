from functools import partial
import utils.Constants as Constants
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from networks.swin_aspp.swin_aspp_3 import build_aspp
from networks.swin_aspp.swin_configs_3 import ASPPConfig
import numpy as np

# 固定函数参数，将relu函数的inplace属性设置为true
nonlinearity = partial(F.relu, inplace=True)


# 级联模块
# 输入(1, 64, 256, 256)

class Model1(nn.Module):
    def __init__(self, channel):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=5, dilation=8, padding=16, bias=False)
        self.conv5 = nn.Conv2d(channel * 4, channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)
        self.bn4 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x1 = self.conv2(x)
        x1 = nonlinearity(self.bn1(x1))

        x2 = self.conv1(self.conv2(x))
        x2 = nonlinearity(self.bn2(x2))

        x3 = self.conv1(self.conv2(self.conv3(x)))
        x3 = nonlinearity(self.bn3(x3))

        x4 = self.conv1(self.conv2(self.conv3(self.conv4(x))))
        x4 = nonlinearity(self.bn4(x4))
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv5(x)
        return x


# 双注意力模块
class DAM(nn.Module):
    def __init__(self, c_size, c_channel, d_channel, is_first=False):
        super(DAM, self).__init__()
        self.is_first = is_first
        self.GPA1 = nn.AdaptiveAvgPool2d((c_size, c_size))
        self.conv1 = nn.Conv2d(in_channels=c_channel, out_channels=1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.sigmoid1 = nn.Sigmoid()

        self.GPA2 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Conv2d(in_channels=c_channel, out_channels=c_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_channel)
        self.conv3 = nn.Conv2d(in_channels=c_channel, out_channels=c_channel, kernel_size=1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv4 = nn.Conv2d(in_channels=d_channel, out_channels=d_channel * 2, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(d_channel * 2)
        self.conv5 = nn.Conv2d(in_channels=c_channel * 2, out_channels=c_channel, kernel_size=1)

    def forward(self, c_x, d_x):
        x1 = self.GPA1(c_x)
        x1 = self.conv1(x1)
        x1 = nonlinearity(self.bn1(x1))
        x1 = self.sigmoid1(x1)
        x1 = torch.mul(c_x, x1)
        x2 = self.GPA2(c_x)
        x2 = self.conv2(x2)
        x2 = nonlinearity(self.bn2(x2))
        x2 = self.conv3(x2)
        x2 = self.sigmoid2(x2)
        x2 = torch.mul(c_x, x2)
        x2 = c_x + x1 + x2
        if (self.is_first == False):
            d_x = self.conv4(d_x)
            d_x = nonlinearity(self.bn4(d_x))
        x = torch.cat((x2, d_x), dim=1)
        x = self.conv5(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.deconv2 = nn.ConvTranspose2d(in_channels, in_channels // 2, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 2)
        self.block1 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(in_channels // 2))
        self.block2 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channels // 2))
        self.block3 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=5, padding=2, bias=False),
                                    nn.BatchNorm2d(in_channels // 2))
        self.conv3 = nn.Conv2d(in_channels * 2, n_filters, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.deconv2(x)
        x = nonlinearity(self.norm2(x))
        x1 = nonlinearity(self.block1(x))
        x2 = nonlinearity(self.block2(x))
        x3 = nonlinearity(self.block3(x))
        x = torch.cat((x, x1, x2, x3), 1)
        x = self.conv3(x)
        x = nonlinearity(self.norm3(x))
        return x


# 上下文模块
class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1)

        self.conv2 = nn.Conv2d(channel * 2, channel, kernel_size=1)

        # 遍历模型的每一层，如果是conv2d或者ConvTranspose2d => 如果偏置不为空的话，将偏置置为0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, d_x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        out = torch.cat((out, d_x), 1)
        out = self.conv2(out)
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=516, out_channels=512, kernel_size=1)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')
        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)
        out = self.conv2(out)
        return out


class DAM_Net_5(nn.Module):
    def __init__(self, img_size=1024, num_classes=Constants.BINARY_CLASS):
        super(DAM_Net_5, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        # resnet.load_state_dict(torch.load('./resnet34.pth'))
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.model1 = Model1(filters[0])

        self.encoder1 = resnet.layer1

        self.encoder2 = resnet.layer2

        self.encoder3 = resnet.layer3

        self.encoder4 = resnet.layer4

        self.dam1 = DAM(img_size // 4, c_channel=filters[0], d_channel=filters[0], is_first=True)
        self.dam2 = DAM(img_size // 8, c_channel=filters[1], d_channel=filters[0])
        self.dam3 = DAM(img_size // 16, c_channel=filters[2], d_channel=filters[1])
        self.dam4 = DAM(img_size // 32, c_channel=filters[3], d_channel=filters[2])

        self.swin_aspp = build_aspp(32, 512, 512, ASPPConfig)

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        m_x = self.model1(x)

        encoder = self.encoder1(x)
        a1 = self.dam1(encoder, m_x)

        encoder = self.encoder2(encoder)

        a2 = self.dam2(encoder, a1)

        encoder = self.encoder3(encoder)
        a3 = self.dam3(encoder, a2)

        encoder = self.encoder4(encoder)
        a4 = self.dam4(encoder, a3)

        # Center
        aspp_x = self.swin_aspp(a4)

        aspp_x = aspp_x + a4
        # # Decoder
        d4 = self.decoder4(aspp_x)
        d4 = d4 + a3
        d3 = self.decoder3(d4)
        d3 = d3 + a2
        d2 = self.decoder2(d3)
        d2 = d2 + a1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


if __name__ == '__main__':
    from torchinfo import summary

    # print(torch.cuda.is_available())
    net = DAM_Net_5()
    summary(net, input_size=(1, 3, 1024, 1024))
