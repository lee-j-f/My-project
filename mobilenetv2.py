import torch
import torch.nn as nn
from models.attention import SEBlock


class InvertedResidual(nn.Module):
    """
    MobileNetV2的倒残差模块（Inverted Residual Block）

    特点：
    1. 先升维（expand）再压缩（project）
    2. 使用深度可分离卷积减少参数量
    3. 在低维度使用线性激活（不使用ReLU）以减少信息损失
    4. 当输入输出维度相同且步长为1时，添加残差连接
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expand_ratio)

        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_Feature_Extractor(nn.Module):
    """
    MobileNetV2_0.5_96 特征提取器

    参数说明：
    - α=0.5（宽度因子）: 所有通道数乘以0.5，大幅减少参数量
    - β对应输入尺寸96（但此处根据脑电输入自适应调整）

    网络结构参考标准MobileNetV2，集成SE注意力模块
    """

    def __init__(self, width_mult=0.5, input_channels=1):
        super(MobileNetV2_Feature_Extractor, self).__init__()

        # MobileNetV2 标准配置: [expand_ratio, output_channels, num_blocks, stride]
        inverted_residual_settings = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # 首层卷积
        first_channel = int(32 * width_mult)
        self.first_conv = nn.Sequential(
            nn.Conv2d(input_channels, first_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(first_channel),
            nn.ReLU6(inplace=True),
        )

        layers = []
        in_channels = first_channel
        for t, c, n, s in inverted_residual_settings:
            out_channels = int(c * width_mult)
            out_channels = max(out_channels, 1)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_channels, out_channels, stride, t))
                in_channels = out_channels
        self.inverted_residuals = nn.Sequential(*layers)

        self.se_block = SEBlock(in_channels, reduction_ratio=4)

        last_channel = int(1280 * width_mult)
        last_channel = max(last_channel, 1)
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.out_features = last_channel

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 32, 128)

        Returns:
            features: (batch, out_features)
        """
        x = self.first_conv(x)          # (B, 16, 16, 64)
        x = self.inverted_residuals(x)   # (B, 160, ...)
        x = self.se_block(x)             # SE注意力重标定
        x = self.last_conv(x)            # (B, 640, ...)
        x = self.avgpool(x)              # (B, 640, 1, 1)
        x = x.view(x.size(0), -1)       # (B, 640)

        return x