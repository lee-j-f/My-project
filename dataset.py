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
        N, C, T = segments.shape
        normalized = np.zeros_like(segments, dtype=np.float32)
        for i in range(N):
            for c in range(C):
                channel_data = segments[i, c, :]
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                if std > 0:
                    normalized[i, c, :] = (channel_data - mean) / std
                else:
                    normalized[i, c, :] = channel_data - mean
        normalized = normalized[:, np.newaxis, :, :]
        return torch.FloatTensor(normalized)

    def _augment(self, x):
        # 1. 高斯噪声（概率40%，幅度0.08）
        if np.random.random() < 0.4:
            noise = torch.randn_like(x) * 0.08
            x = x + noise

        # 2. 时间平移（概率30%）
        if np.random.random() < 0.3:
            shift = np.random.randint(-8, 8)
            x = torch.roll(x, shifts=shift, dims=-1)

        # 3. 通道随机遮蔽（概率15%，遮1-2个）
        if np.random.random() < 0.15:
            n_mask = np.random.randint(1, 3)
            channels = np.random.choice(32, n_mask, replace=False)
            x[0, channels, :] = 0

        # 4. 幅值缩放（概率20%）
        if np.random.random() < 0.2:
            scale = np.random.uniform(0.9, 1.1)
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