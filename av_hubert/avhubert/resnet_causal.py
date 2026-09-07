# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

import logging
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys as _sys
if len(_sys.argv) == 1:
    from resnet import ResNet
else:
    from .resnet import ResNet


logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def downsample_basic_block(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def downsample_basic_block_v2(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.AvgPool2d(
            kernel_size=stride,
            stride=stride,
            ceil_mode=True,
            count_include_pad=False,
        ),
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(outplanes),
    )


class CausalConv3d(nn.Module):
    """
    Causal in time, symmetric in space.

    Input:  [B, C, T, H, W]
    Output: [B, C_out, T_out, H_out, W_out]

    Only uses frames <= current frame.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        groups=1,
        bias=False,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,   # we do padding manually
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        kt, kh, kw = self.kernel_size
        dt, dh, dw = self.dilation

        # causal padding in time
        pad_t_left = (kt - 1) * dt
        pad_t_right = 0

        # symmetric padding in space
        pad_h = ((kh - 1) * dh) // 2
        pad_w = ((kw - 1) * dw) // 2

        # F.pad for 5D uses:
        # (W_left, W_right, H_left, H_right, T_left, T_right)
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_t_left, pad_t_right))
        return self.conv(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type='relu'):
        super(BasicBlock, self).__init__()

        assert relu_type in ['relu', 'prelu']

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        else:
            raise Exception('relu type not implemented')

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class CausalResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        relu_type='relu',
        gamma_zero=False,
        avg_pool_downsample=False,
    ):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = (
            downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block
        )

        super(CausalResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv3d):
                kt, kh, kw = m.kernel_size
                n = kt * kh * kw * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes * block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                relu_type=self.relu_type,
            )
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class CausalResEncoder(nn.Module):
    def __init__(self, relu_type, weights):
        super(CausalResEncoder, self).__init__()
        self.frontend_nout = 64
        self.backend_out = 512

        frontend_relu = (
            nn.PReLU(num_parameters=self.frontend_nout)
            if relu_type == 'prelu'
            else nn.ReLU(inplace=True)
        )

        # Only this block needed to change to become causal in time
        self.frontend3D = nn.Sequential(
            CausalConv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )

        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        if weights is not None:
            logger.info(f"Load {weights} for resnet")
            std = torch.load(weights, map_location=torch.device('cpu'))['model_state_dict']
            frontend_std, trunk_std = OrderedDict(), OrderedDict()

            for key, val in std.items():
                new_key = '.'.join(key.split('.')[1:])
                if 'frontend3D' in key:
                    frontend_std[new_key] = val
                if 'trunk' in key:
                    trunk_std[new_key] = val

            # old pretrained weights for Conv3d still load fine into CausalConv3d.conv
            remapped_frontend_std = OrderedDict()
            for key, val in frontend_std.items():
                if key.startswith("frontend3D.0."):
                    # old: frontend3D.0.weight
                    # new: frontend3D.0.conv.weight
                    new_key = key.replace("frontend3D.0.", "0.conv.")
                else:
                    # after stripping module prefix above, keys are already local to frontend3D
                    # for Sequential entries they should look like 1.xxx, 2.weight, ...
                    new_key = key.replace("frontend3D.", "")
                remapped_frontend_std[new_key] = val

            try:
                self.frontend3D.load_state_dict(remapped_frontend_std, strict=False)
            except Exception:
                logger.warning("Could not fully load frontend3D weights; loading trunk only.")

            self.trunk.load_state_dict(trunk_std, strict=False)

    def forward(self, x):
        # x: [B, 1, T, H, W]
        B, C, T, H, W = x.size()

        x = self.frontend3D(x)      # [B, C', T', H', W']
        Tnew = x.shape[2]

        x = self.threeD_to_2D_tensor(x)   # [B*T', C', H', W']
        x = self.trunk(x)                 # [B*T', 512]

        x = x.view(B, Tnew, x.size(1))
        x = x.transpose(1, 2).contiguous()   # [B, 512, T']
        return x

    def threeD_to_2D_tensor(self, x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.reshape(n_batch * s_time, n_channels, sx, sy)