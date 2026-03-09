import numpy as np
import scipy.io as sio
import os


class DEAPDataLoader:
    """
    DEAP数据集加载器
    - 加载32个受试者的预处理脑电数据
    - 去除基线信号并进行滑动窗口分割
    """

    def __init__(self, data_dir, num_channels=32, sampling_rate=128,
                 baseline_duration=3, trial_duration=60, window_size=1):
        """
        Args:
            data_dir: DEAP数据集目录路径
            num_channels: 使用的脑电通道数（默认32）
            sampling_rate: 采样率（Hz），DEAP预处理版本为128Hz
            baseline_duration: 基线信号时长（秒）
            trial_duration: 实验信号时长（秒）
            window_size: 滑动窗口大小（秒）
        """
        self.data_dir = data_dir
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.baseline_duration = baseline_duration
        self.trial_duration = trial_duration
        self.window_size = window_size

        self.baseline_points = baseline_duration * sampling_rate  # 3*128 = 384
        self.trial_points = trial_duration * sampling_rate        # 60*128 = 7680
        self.window_points = window_size * sampling_rate          # 1*128 = 128

        # 32通道名称（国际10-20系统）
        self.channel_names = [
            'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
            'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
            'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
        ]

    def load_subject(self, subject_id):
        """
        加载单个受试者的数据

        Args:
            subject_id: 受试者编号（1-32）

        Returns:
            data: shape (40, 40, 8064) 的原始数据
            labels: shape (40, 4) 的标签 [valence, arousal, dominance, liking]
        """
        filename = f's{subject_id:02d}.dat'
        filepath = os.path.join(self.data_dir, filename)

        # 使用pickle加载.dat文件（DEAP预处理版本格式）
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        eeg_data = data['data']     # (40, 40, 8064)
        labels = data['labels']     # (40, 4)

        return eeg_data, labels

    def remove_baseline(self, eeg_data):
        """
        去除基线信号：用实验信号减去基线信号的均值
        消除静息脑电对情绪诱发脑电的影响

        Args:
            eeg_data: shape (40, 32, 8064) - 仅脑电通道

        Returns:
            corrected_data: shape (40, 32, 7680) - 去除基线后的实验信号
        """
        # 提取基线信号（前3秒，384个采样点）
        baseline = eeg_data[:, :, :self.baseline_points]  # (40, 32, 384)
        # 计算基线均值
        baseline_mean = np.mean(baseline, axis=2, keepdims=True)  # (40, 32, 1)

        # 提取实验信号（后60秒，7680个采样点）
        trial_data = eeg_data[:, :, self.baseline_points:]  # (40, 32, 7680)

        # 去除基线：实验信号 - 基线均值
        corrected_data = trial_data - baseline_mean

        return corrected_data

    def sliding_window_segmentation(self, corrected_data):
        """
        滑动窗口分割：将60秒信号分割为1秒的片段
        每个片段作为独立样本，增加训练数据量

        Args:
            corrected_data: shape (40, 32, 7680)

        Returns:
            segments: shape (40*60, 32, 128) = (2400, 32, 128)
        """
        num_trials = corrected_data.shape[0]
        num_segments_per_trial = self.trial_points // self.window_points  # 7680/128 = 60

        segments = []
        for trial_idx in range(num_trials):
            for seg_idx in range(num_segments_per_trial):
                start = seg_idx * self.window_points
                end = start + self.window_points
                segment = corrected_data[trial_idx, :, start:end]  # (32, 128)
                segments.append(segment)

        segments = np.array(segments)  # (2400, 32, 128)
        return segments

    def prepare_labels(self, labels, threshold=5.0):
        """
        将连续标签转换为二分类标签
        阈值设为5.0：>=5为高（1），<5为低（0）

        Args:
            labels: shape (40, 4) - [valence, arousal, dominance, liking]
            threshold: 分类阈值

        Returns:
            binary_labels: dict，包含每个维度的二分类标签
                每个维度: shape (2400,) - 与分割后的片段一一对应
        """
        num_segments_per_trial = self.trial_points // self.window_points  # 60

        binary_labels = {}
        dimensions = ['valence', 'arousal', 'dominance', 'liking']

        for dim_idx, dim_name in enumerate(dimensions):
            dim_labels = labels[:, dim_idx]  # (40,)
            # 二分类：>=threshold为高(1)，<threshold为低(0)
            binary = (dim_labels >= threshold).astype(np.int64)  # (40,)
            # 扩展标签：每个trial的60个片段共享同一标签
            expanded = np.repeat(binary, num_segments_per_trial)  # (2400,)
            binary_labels[dim_name] = expanded

        return binary_labels

    def preprocess_subject(self, subject_id, threshold=5.0):
        """
        对单个受试者执行完整预处理流程

        Returns:
            segments: (2400, 32, 128) 预处理后的脑电片段
            binary_labels: dict，各维度的二分类标签
        """
        # 1. 加载数据
        eeg_data, labels = self.load_subject(subject_id)

        # 2. 仅保留32通道脑电信号
        eeg_data = eeg_data[:, :self.num_channels, :]  # (40, 32, 8064)

        # 3. 去除基线
        corrected_data = self.remove_baseline(eeg_data)  # (40, 32, 7680)

        # 4. 滑动窗口分割
        segments = self.sliding_window_segmentation(corrected_data)  # (2400, 32, 128)

        # 5. 准备二分类标签
        binary_labels = self.prepare_labels(labels, threshold)

        return segments, binary_labels

    def preprocess_all(self, threshold=5.0):
        """
        预处理所有32个受试者的数据

        Returns:
            all_segments: (32*2400, 32, 128) = (76800, 32, 128)
            all_labels: dict，各维度的二分类标签
        """
        all_segments = []
        all_labels = {dim: [] for dim in ['valence', 'arousal', 'dominance', 'liking']}

        for subject_id in range(1, 33):
            print(f"Processing subject s{subject_id:02d}...")
            segments, binary_labels = self.preprocess_subject(subject_id, threshold)
            all_segments.append(segments)
            for dim in all_labels:
                all_labels[dim].append(binary_labels[dim])

        all_segments = np.concatenate(all_segments, axis=0)
        for dim in all_labels:
            all_labels[dim] = np.concatenate(all_labels[dim], axis=0)

        print(f"Total segments: {all_segments.shape}")
        return all_segments, all_labels