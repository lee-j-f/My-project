import torch
import torch.nn as nn
from models.capsule_network import EmotionCapsuleNet


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积（MobileNetV2核心组件）
    将标准卷积分解为：
    1. 深度卷积（Depthwise）：每个通道独立卷积
    2. 逐点卷积（Pointwise）：1x1卷积融合通道信息
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class InvertedResidual(nn.Module):
    """
    倒残差模块
    结构：1x1升维 → 3x3深度卷积 → 1x1降维
    当stride=1且输入输出通道相同时，使用残差连接
    """

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

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
        return self.conv(x)


class SEBlock(nn.Module):
    """
    SE通道注意力模块
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * weight


class EEGMobileNetV2(nn.Module):
    """
    借鉴MobileNetV2架构的脑电特征提取网络
    保留MobileNetV2的核心设计：
    - 深度可分离卷积（减少参数量）
    - 倒残差结构（减少信息丢失）
    - 线性瓶颈（低维不用ReLU）
    针对脑电信号的适配：
    - 更少的层数和通道数（适配小数据集）
    - 输入为1通道 (1, 32, 128)
    - 加入SE通道注意力
    - 加入Dropout防过拟合
    """

    def __init__(self, dropout_rate=0.5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        # 倒残差模块堆叠
        self.blocks = nn.Sequential(
            # Block 1: 16→24, 下采样
            InvertedResidual(16, 24, stride=2, expand_ratio=2),
            InvertedResidual(24, 24, stride=1, expand_ratio=2),
            nn.Dropout2d(0.2),

            # Block 2: 24→32, 下采样
            InvertedResidual(24, 32, stride=2, expand_ratio=2),
            InvertedResidual(32, 32, stride=1, expand_ratio=2),
            nn.Dropout2d(0.2),

            # Block 3: 32→64, 下采样
            InvertedResidual(32, 64, stride=2, expand_ratio=2),
            InvertedResidual(64, 64, stride=1, expand_ratio=2),
            nn.Dropout2d(0.3),

            # Block 4: 64→96
            InvertedResidual(64, 96, stride=1, expand_ratio=2),
            nn.Dropout2d(0.3),
        )

        # SE通道注意力
        self.se = SEBlock(96, reduction=4)

        # 最终卷积 + 全局池化
        self.head = nn.Sequential(
            nn.Conv2d(96, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
        )

        self.out_features = 128

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.se(x)
        x = self.head(x)
        return x


class EEGEmotionRecognitionModel(nn.Module):
    """
    基于SE-MobileNetV2与胶囊网络的脑电情绪识别模型

    特征提取：借鉴MobileNetV2的深度可分离卷积和倒残差结构
    通道注意力：SE模块自适应增强情绪相关脑电通道
    分类器：双层胶囊网络 + 动态路由
    """

    def __init__(self, num_classes=2, width_mult=0.5,
                 primary_caps=16, primary_dim=8,
                 emotion_dim=16, routing_iterations=3,
                 dropout_rate=0.5, **kwargs):
        super().__init__()

        self.feature_extractor = EEGMobileNetV2(dropout_rate=dropout_rate)

        self.classifier = EmotionCapsuleNet(
            in_features=self.feature_extractor.out_features,
            num_classes=num_classes,
            primary_caps=primary_caps,
            primary_dim=primary_dim,
            emotion_dim=emotion_dim,
            routing_iterations=routing_iterations
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        capsule_output, probs = self.classifier(features)
        return capsule_output, probs

    def get_param_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'trainable_ratio': trainable / total
        }