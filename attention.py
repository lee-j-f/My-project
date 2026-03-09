import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) 通道注意力模块

    1. Squeeze: 全局平均池化，将空间维度压缩为标量
    2. Excitation: 通过两个全连接层学习通道间的依赖关系
    3. Reweight: 将学习到的通道权重乘回原特征图
    """

    def __init__(self, in_channels, reduction_ratio=16):

        super(SEBlock, self).__init__()

        reduced_channels = max(in_channels // reduction_ratio, 1)

        # Squeeze
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        batch_size, channels, _, _ = x.size()

        # Squeeze: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        y = self.squeeze(x).view(batch_size, channels)

        # Excitation: (B, C) -> (B, C/r) -> (B, C)
        y = self.excitation(y).view(batch_size, channels, 1, 1)

        return x * y.expand_as(x)