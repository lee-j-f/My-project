import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

HAS_CUDA = torch.cuda.is_available()


class EEGDataset(Dataset):

    def __init__(self, segments, labels, augment=False):
        self.segments = self._normalize(segments)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def _normalize(self, segments):
        """全局通道标准化：用该通道所有样本的统计量"""
        N, C, T = segments.shape
        normalized = np.zeros_like(segments, dtype=np.float32)
        for c in range(C):
            channel_all = segments[:, c, :].flatten()
            mean = np.mean(channel_all)
            std = np.std(channel_all)
            if std > 0:
                normalized[:, c, :] = (segments[:, c, :] - mean) / std
            else:
                normalized[:, c, :] = segments[:, c, :] - mean
        normalized = normalized[:, np.newaxis, :, :]
        return torch.FloatTensor(normalized)

    def _augment(self, x):
        # 1. 高斯噪声（概率30%，幅度0.03）
        if np.random.random() < 0.3:
            noise = torch.randn_like(x) * 0.03
            x = x + noise

        # 2. 时间平移（概率20%，范围±3）
        if np.random.random() < 0.2:
            shift = np.random.randint(-3, 4)
            x = torch.roll(x, shifts=shift, dims=-1)

        # 3. 通道随机遮蔽（概率10%，遮1个）
        if np.random.random() < 0.1:
            channel = np.random.randint(0, 32)
            x[0, channel, :] = 0

        # 4. 幅值缩放（概率15%，范围0.95~1.05）
        if np.random.random() < 0.15:
            scale = np.random.uniform(0.95, 1.05)
            x = x * scale

        return x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.segments[idx].clone()
        if self.augment:
            x = self._augment(x)
        return x, self.labels[idx]


def create_subject_dependent_loaders(segments, labels, batch_size=64,
                                      n_splits=10, fold_idx=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(kf.split(segments))
    train_idx, test_idx = splits[fold_idx]

    train_dataset = EEGDataset(segments[train_idx], labels[train_idx], augment=True)
    test_dataset = EEGDataset(segments[test_idx], labels[test_idx], augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2 if HAS_CUDA else 0, pin_memory=HAS_CUDA
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2 if HAS_CUDA else 0, pin_memory=HAS_CUDA
    )

    return train_loader, test_loader


def create_trial_based_loaders(segments, labels, batch_size=64,
                                n_splits=10, fold_idx=0,
                                num_trials=40, segments_per_trial=60):
    """按trial级别划分的十折交叉验证，避免数据泄漏"""
    expected_len = num_trials * segments_per_trial
    if len(segments) != expected_len:
        raise ValueError(
            f"segments length {len(segments)} does not match "
            f"num_trials ({num_trials}) * segments_per_trial ({segments_per_trial}) = {expected_len}"
        )
    trial_indices = np.arange(num_trials)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(kf.split(trial_indices))
    train_trials, test_trials = splits[fold_idx]

    train_idx = []
    test_idx = []
    for t in train_trials:
        start = t * segments_per_trial
        end = start + segments_per_trial
        train_idx.extend(range(start, end))
    for t in test_trials:
        start = t * segments_per_trial
        end = start + segments_per_trial
        test_idx.extend(range(start, end))

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    train_dataset = EEGDataset(segments[train_idx], labels[train_idx], augment=True)
    test_dataset = EEGDataset(segments[test_idx], labels[test_idx], augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2 if HAS_CUDA else 0, pin_memory=HAS_CUDA
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2 if HAS_CUDA else 0, pin_memory=HAS_CUDA
    )
    return train_loader, test_loader