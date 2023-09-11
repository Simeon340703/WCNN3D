"""This code is a pyTorch implementation of WCNN3D paper. https://www.mdpi.com/1424-8220/22/18/7010
@c Simegnew Alaba
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchplus.nn import Empty, GroupNorm, Sequential
import numpy as np

def conv2D(in_channels, out_channels, kernel_size=3, bias=True, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2)+dilation-1, bias=bias, dilation=dilation)


def dwt_init(x, name='haar'):
    x[:, :, 0::2, :] /= 2
    x[:, :, 1::2, :] /= 2
    x_LL, x_HL, x_LH, x_HH = None, None, None, None

    if name == 'haar':
        # Haar wavelets
        x1 = x[:, :, :-1:2, :-1:2]
        x2 = x[:, :, :-1:2, 1::2]
        x3 = x[:, :, 1::2, :-1:2]
        x4 = x[:, :, 1::2, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

    elif name == 'db4':
        # Daubechies 4 wavelets
        c = np.array([(1 + np.sqrt(3)) / (4 * np.sqrt(2)), (3 + np.sqrt(3)) / (4 * np.sqrt(2)),
                      (3 - np.sqrt(3)) / (4 * np.sqrt(2)), (1 - np.sqrt(3)) / (4 * np.sqrt(2))])
        x1 = x[:, :, :-1:2, :-1:2]
        x2 = x[:, :, :-1:2, 1::2]
        x3 = x[:, :, 1::2, :-1:2]
        x4 = x[:, :, 1::2, 1::2]
        x_LL = c[0] * x1 + c[1] * x2 + c[2] * x3 + c[3] * x4
        x_HL = -c[0] * x1 - c[1] * x2 + c[2] * x3 + c[3] * x4
        x_LH = -c[0] * x1 + c[1] * x2 - c[2] * x3 + c[3] * x4
        x_HH = c[0] * x1 - c[1] * x2 - c[2] * x3

    return torch.cat((x_LL, x_HL, x_LH, x_HH), dim=1)


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


def iwt_init(x, name='haar'):
    r = 2
    in_batch, in_channel, in_height, in_width = x.shape
    out_batch = in_batch
    out_channel = in_channel // (r ** 2)
    out_height = r * in_height
    out_width = r * in_width

    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:, out_channel:2 * out_channel, :, :] / 2
    x3 = x[:, 2 * out_channel:3 * out_channel, :, :] / 2
    x4 = x[:, 3 * out_channel:4 * out_channel, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width], dtype=torch.float32, device=x.device)

    if name == 'haar':
        # Haar wavelet
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    elif name == 'db4':
        # Db4
        c = torch.tensor([1 + np.sqrt(3), 3 + np.sqrt(3), 3 - np.sqrt(3), 1 - np.sqrt(3)], device=x.device)
        c = c / (2 * np.sqrt(2))

        h[:, :, 0::2, 0::2] = c[0] * x1 - c[1] * x2 - c[2] * x3 + c[3] * x4
        h[:, :, 1::2, 0::2] = c[0] * x1 - c[1] * x2 + c[2] * x3 - c[3] * x4
        h[:, :, 0::2, 1::2] = c[0] * x1 + c[1] * x2 - c[2] * x3 - c[3] * x4
        h[:, :, 1::2, 1::2] = c[0] * x1 + c[1] * x2 + c[2] * x3 + c[3] * x4
    else:
        raise ValueError("Unsupported wavelet name: %s" % name)

    return h

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class BasicBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=False):
        super(BasicBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)

        if self.bn is not None:
            out = self.bn(out)

        if self.act is not None:
            out = self.act(out)

        return out


class EBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, conv=conv2D, bias=True, bn=False, res_scale=1):
        super(EBlock, self).__init__()

        self.conv = conv(in_channels, out_channels, kernel_size, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95) if bn else None
        self.act = nn.ReLU(inplace=True)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.act(out).mul(self.res_scale)

        return out


class DBlock(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size=3, conv=conv2D,
                 bias=True, bn=False,  res_scale=1):
        super(DBlock, self).__init__()

        # Define convolutional layers with dilation
        self.conv1 = conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2)
        self.conv2 = conv(out_channels, out_channels, kernel_size, bias=bias, dilation=1)

        # Define batch normalization and activation layers
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95) if bn else None
        self.act = nn.ReLU(inplace=True)

        self.res_scale = res_scale

    def forward(self, x):
        # Apply convolutional layers and activation function
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        if self.bn is not None:
            out = self.bn(out)
        out = self.act(out)

        # Scale the residual connection
        out = out.mul(self.res_scale)

        return out

class Conv1x1Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, bias=False):
        super(Conv1x1Basic, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
    def forward(self, x):
        return self.conv1(x)

        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class WcnnRPNCar(nn.Module):
    def __init__(self, nChannels=64):
        super(WcnnRPNCar, self).__init__()
        self.DWT = DWT()
        self.IWT = IWT()

        conv = [EBlock(nChannels, nChannels)]
        self.conv0 = Sequential(*conv)

        def make_layer(in_planes, out_planes, num_blocks=1):
            layers = []
            for _ in range(num_blocks):
                layers.append(DBlock(in_planes, out_planes))
                layers.append(EBlock(out_planes, out_planes))
                in_planes = out_planes
            return Sequential(*layers)

        # Encoder layers
        self.e1 = DBlock(nChannels, nChannels)
        self.e2 = make_layer(nChannels, nChannels)
        self.e3 = make_layer(nChannels, nChannels)
        self.e4 = make_layer(nChannels * 2, nChannels * 2)
        # layers between dwt and iwt
        self.ed = Sequential(
            EBlock(nChannels * 4, nChannels * 2),
            DBlock(nChannels * 2, nChannels * 2),
            DBlock(nChannels * 2, nChannels * 2),
            EBlock(nChannels * 2, nChannels * 4)
        )

        # Decoder layers
        self.i1 = DBlock(nChannels, nChannels)
        self.i2 = Sequential(
            DBlock(nChannels, nChannels),
            EBlock(nChannels, nChannels * 4)
        )
        self.i3 = Sequential(
            DBlock(nChannels * 2, nChannels * 2),
            EBlock(nChannels * 2, nChannels * 4)
        )
        self.i4 = Sequential(
            DBlock(nChannels * 4, nChannels * 4),
            EBlock(nChannels * 4, nChannels * 8)
        )

        self.tail = conv2D(nChannels, nChannels)
        # 1x1 Conv layers
        self.conv1 = Conv1x1Basic(nChannels, 16)
        self.conv2 = Conv1x1Basic(nChannels, 32)
        self.conv3 = Conv1x1Basic(nChannels * 2, 64)
        self.conv4 = Conv1x1Basic(nChannels * 4, 512)
        self.conv5 = Conv1x1Basic(nChannels * 8, nChannels * 4)

    def forward(self, x, bev=None):
        x0 = self.e1(self.conv0(x))
        x0 = self.e2(x0) + x0
        x1 = self.DWT(self.conv1(x0))

        x2 = self.e3(x1) + x1
        x2 = self.DWT(self.conv2(x2))

        x3 = self.e4(x2) + x2
        x3 = self.DWT(self.conv3(x3))
        x_ = self.conv4(self.ed(x3))
        x_ = self.conv5(self.IWT(self.DWT(x_))) + x3

        x_ = self.IWT(self.i4(x_)) + x2
        x_ = self.IWT(self.i3(x_)) + x1
        x = self.tail(self.i1(x_))

        return x

class WcnnRPNPedCycle(nn.Module):
    def __init__(self, nChannels=64):
        super(WcnnRPNPedCycle, self).__init__()
        self.nChannels = nChannels
        self.DWT = DWT()
        self.IWT = IWT()

        conv = [EBlock(nChannels, nChannels)]
        self.conv0 = Sequential(*conv)

        def make_layer(in_planes, out_planes, num_blocks=1):
            layers = []
            for _ in range(num_blocks):
                layers.append(DBlock(in_planes, out_planes))
                layers.append(EBlock(out_planes, out_planes))
                in_planes = out_planes
            return Sequential(*layers)

        # Encoder layers
        self.e1 = DBlock(nChannels, nChannels)
        self.e2 = make_layer(nChannels, nChannels)
        self.e3 = make_layer(nChannels, nChannels)
        self.e4 = make_layer(nChannels * 2, nChannels * 2)
        # layers between dwt and iwt
        self.ed = Sequential(
            EBlock(nChannels * 2, nChannels * 2),
            DBlock(nChannels * 2, nChannels * 2),
            DBlock(nChannels * 2, nChannels * 2),
            EBlock(nChannels * 2, nChannels * 2)
        )

        # Decoder layers
        self.i1 = DBlock(nChannels, nChannels)
        self.i2 = Sequential(
            DBlock(nChannels, nChannels),
            EBlock(nChannels, nChannels * 4)
        )
        self.i3 = Sequential(
            DBlock(nChannels, nChannels),
            EBlock(nChannels, nChannels * 4)
        )
        self.i4 = Sequential(
            DBlock(nChannels * 2, nChannels * 2),
            EBlock(nChannels * 2, nChannels * 4)
        )

        self.tail = conv2D(nChannels, nChannels)
        # 1x1 Conv layers
        self.conv1 = Conv1x1Basic(nChannels, 16)
        self.conv2 = Conv1x1Basic(nChannels, 32)
        self.conv3 = Conv1x1Basic(nChannels * 2, 64)
        self.conv4 = Conv1x1Basic(nChannels * 2, 256)
        self.conv5 = Conv1x1Basic(nChannels * 4, nChannels * 2)

    def forward(self, x, bev=None):
        x0 = self.e1(self.conv0(x))
        x0 = self.e2(x0) + x0
        x1 = self.DWT(self.conv1(x0))

        x2 = self.e3(x1) + x1
        x2 = self.DWT(self.conv2(x2))
        x_ = self.conv4(self.ed(x2))
        x_ = self.conv5(self.IWT(self.DWT(x_))) + x2

        x_ = self.IWT(self.i4(x_)) + x1
        x_ = self.IWT(self.i3(x_)) + x0
        x = self.tail(self.i1(x_)) + x0

        return x
