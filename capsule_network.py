import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(tensor, dim=-1):

    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    # clamp 防止极端值
    squared_norm = torch.clamp(squared_norm, min=1e-8, max=1e4)
    norm = torch.sqrt(squared_norm)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * (tensor / (norm + 1e-8))


class PrimaryCapsules(nn.Module):
    """初级胶囊层"""

    def __init__(self, in_features, num_capsules=16, capsule_dim=8):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

        self.fc = nn.Sequential(
            nn.Linear(in_features, num_capsules * capsule_dim),
            nn.LayerNorm(num_capsules * capsule_dim),
        )

        nn.init.xavier_normal_(self.fc[0].weight)
        nn.init.zeros_(self.fc[0].bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc(x)
        x = x.view(batch_size, self.num_capsules, self.capsule_dim)
        x = squash(x, dim=-1)
        return x


class DynamicRouting(nn.Module):
    """数值稳定的动态路由"""

    def __init__(self, num_capsules_in, num_capsules_out,
                 capsule_dim_in=8, capsule_dim_out=16,
                 num_iterations=3):
        super().__init__()
        self.num_capsules_in = num_capsules_in
        self.num_capsules_out = num_capsules_out
        self.num_iterations = num_iterations
        self.W = nn.Parameter(
            torch.randn(num_capsules_in, num_capsules_out,
                        capsule_dim_out, capsule_dim_in) * 0.005
        )

    def forward(self, u):
        batch_size = u.size(0)
        u_expand = u.unsqueeze(2).unsqueeze(4)
        W_expand = self.W.unsqueeze(0)
        u_hat = torch.matmul(W_expand, u_expand).squeeze(-1)

        u_hat = torch.clamp(u_hat, min=-10.0, max=10.0)

        b = torch.zeros(batch_size, self.num_capsules_in,
                        self.num_capsules_out, device=u.device)

        for iteration in range(self.num_iterations):
            c = F.softmax(b, dim=2)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
            v = squash(s, dim=-1)

            if iteration < self.num_iterations - 1:
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)
                # clamp agreement 防止 b 爆炸
                agreement = torch.clamp(agreement, min=-10.0, max=10.0)
                b = b + agreement

        return v


class EmotionCapsuleNet(nn.Module):
    """双层胶囊网络"""

    def __init__(self, in_features, num_classes=2, primary_caps=16,
                 primary_dim=8, emotion_dim=16, routing_iterations=3):
        super().__init__()
        self.num_classes = num_classes

        self.primary_capsules = PrimaryCapsules(
            in_features=in_features,
            num_capsules=primary_caps,
            capsule_dim=primary_dim
        )

        self.routing = DynamicRouting(
            num_capsules_in=primary_caps,
            num_capsules_out=num_classes,
            capsule_dim_in=primary_dim,
            capsule_dim_out=emotion_dim,
            num_iterations=routing_iterations
        )

    def forward(self, features):
        primary_caps = self.primary_capsules(features)
        v = self.routing(primary_caps)
        probs = torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)
        probs = torch.clamp(probs, min=0.0, max=0.9999)
        return v, probs