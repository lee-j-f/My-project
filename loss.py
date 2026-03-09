import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):
    """
    数值稳定的 MarginLoss
    """

    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_val=0.5):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_val = lambda_val

    def forward(self, probs, labels):
        # 严格限制范围
        probs = torch.clamp(probs, min=0.0, max=0.9999)

        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        positive_loss = one_hot * F.relu(self.m_plus - probs) ** 2
        negative_loss = self.lambda_val * (1 - one_hot) * F.relu(probs - self.m_minus) ** 2

        loss = (positive_loss + negative_loss).sum(dim=-1).mean()

        return loss