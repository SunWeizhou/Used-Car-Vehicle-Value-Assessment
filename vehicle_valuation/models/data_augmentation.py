"""
数据增强模块

为Transformer故障预测模型提供数据增强功能。

主要功能：
1. 时间序列噪声添加
2. 滑动窗口大小变化
3. 步长变化
4. 多样化数据生成

作者：Claude AI
创建时间：2026-02-01
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
import random

class DataAugmenter:
    """数据增强器"""

    def __init__(self, config: Dict):
        self.config = config
        # 获取augmentation配置，可能是嵌套在data中
        if 'data' in config and 'augmentation' in config['data']:
            self.augmentation_config = config['data']['augmentation']
        else:
            self.augmentation_config = config.get('augmentation', {})
        self.enabled = self.augmentation_config.get('enabled', False)
        self.noise_level = self.augmentation_config.get('noise_level', 0.01)
        self.noise_types = self.augmentation_config.get('noise_types', ['gaussian'])
        self.window_sizes = self.augmentation_config.get('window_sizes', [5, 6, 7])
        self.strides = self.augmentation_config.get('strides', [2, 3, 4])
        self.augmentation_factor = self.augmentation_config.get('augmentation_factor', 2)

        # 创建scaler用于标准化
        self.temporal_scaler = StandardScaler()
        self.system_scaler = StandardScaler()

        logging.info(f"数据增强器初始化完成 - enabled: {self.enabled}")
        if self.enabled:
            logging.info(f"配置: noise_level={self.noise_level}, window_sizes={self.window_sizes}")
            logging.info(f"配置: strides={self.strides}, augmentation_factor={self.augmentation_factor}")

    def add_gaussian_noise(self, data: pd.DataFrame, noise_level: float) -> pd.DataFrame:
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_level, data.shape)
        augmented_data = data.copy()
        augmented_data.iloc[:, :] = data.values + noise
        return augmented_data

    def add_uniform_noise(self, data: pd.DataFrame, noise_level: float) -> pd.DataFrame:
        """添加均匀分布噪声"""
        noise = np.random.uniform(-noise_level, noise_level, data.shape)
        augmented_data = data.copy()
        augmented_data.iloc[:, :] = data.values + noise
        return augmented_data

    def scale_features(self, data: pd.DataFrame, scale_range: Tuple[float, float]) -> pd.DataFrame:
        """特征缩放增强"""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        augmented_data = data.copy()
        augmented_data.iloc[:, :] = data.values * scale_factor
        return augmented_data

    def create_windows_with_different_sizes(self, df: pd.DataFrame, base_window_size: int) -> List[pd.DataFrame]:
        """创建不同大小的滑动窗口"""
        windows = []

        # 基础窗口
        for size in self.window_sizes:
            if size <= len(df):
                window = df.iloc[-size:].copy()
                windows.append(window)

        return windows

    def create_sequences_with_different_strides(self, df: pd.DataFrame, base_stride: int) -> List[pd.DataFrame]:
        """创建不同步长的序列"""
        sequences = []

        for stride in self.strides:
            if stride < len(df):
                # 从不同起始点创建序列
                for start_idx in range(0, len(df) - 6, stride):
                    end_idx = min(start_idx + 6, len(df))
                    sequence = df.iloc[start_idx:end_idx].copy()
                    sequences.append(sequence)

        return sequences

    def augment_single_sequence(self, sequence: pd.DataFrame) -> List[pd.DataFrame]:
        """增强单个序列"""
        if not self.enabled:
            return [sequence]

        augmented_sequences = [sequence]

        # 生成多个增强版本
        for _ in range(self.augmentation_factor - 1):
            aug_seq = sequence.copy()

            # 随机选择噪声类型
            if random.choice(self.noise_types) == 'gaussian':
                aug_seq = self.add_gaussian_noise(aug_seq, self.noise_level)
            else:
                aug_seq = self.add_uniform_noise(aug_seq, self.noise_level)

            # 随机缩放某些特征
            if random.random() < 0.3:  # 30%概率进行特征缩放
                scale_range = (0.9, 1.1)
                aug_seq = self.scale_features(aug_seq, scale_range)

            augmented_sequences.append(aug_seq)

        return augmented_sequences

    def augment_dataset(self, sequences: List[Dict]) -> List[Dict]:
        """增强整个数据集"""
        if not self.enabled:
            logging.info("数据增强未启用，返回原始数据")
            return sequences

        logging.info(f"开始数据增强 - 原始序列数: {len(sequences)}")

        augmented_sequences = []

        for i, seq in enumerate(sequences):
            # 增强当前序列
            augmented_seqs = self.augment_single_sequence(seq['window_data'])

            # 为每个增强版本创建新的序列
            for j, aug_data in enumerate(augmented_seqs):
                new_seq = {
                    'vin': f"{seq['vin']}_aug_{i}_{j}",
                    'window_start': seq['window_start'],
                    'window_end': seq['window_end'],
                    'window_data': aug_data,
                    'labels': seq['labels'],
                    'is_augmented': True if j > 0 else False
                }
                augmented_sequences.append(new_seq)

        logging.info(f"数据增强完成 - 增强后序列数: {len(augmented_sequences)}")
        logging.info(f"增强倍数: {len(augmented_sequences) / len(sequences):.1f}x")

        return augmented_sequences

    def get_augmentation_stats(self, sequences: List[Dict]) -> Dict:
        """获取增强统计信息"""
        stats = {
            'total_sequences': len(sequences),
            'original_sequences': sum(1 for seq in sequences if not seq.get('is_augmented', False)),
            'augmented_sequences': sum(1 for seq in sequences if seq.get('is_augmented', False)),
            'augmentation_factor': len(sequences) / max(1, sum(1 for seq in sequences if not seq.get('is_augmented', False)))
        }

        # 计算序列长度分布
        lengths = [len(seq['window_data']) for seq in sequences]
        stats['sequence_length_stats'] = {
            'min': min(lengths),
            'max': max(lengths),
            'mean': np.mean(lengths),
            'std': np.std(lengths)
        }

        return stats

    def create_mixed_windows(self, df: pd.DataFrame, base_window_size: int) -> List[pd.DataFrame]:
        """创建混合窗口（结合不同窗口大小和步长）"""
        mixed_windows = []

        # 基础窗口
        base_window = df.iloc[-base_window_size:].copy()
        mixed_windows.append(base_window)

        # 不同大小的窗口
        for size in self.window_sizes:
            if size != base_window_size and size <= len(df):
                window = df.iloc[-size:].copy()
                mixed_windows.append(window)

        # 不同步长的窗口
        for stride in self.strides:
            if stride < len(df) - base_window_size:
                for start_idx in range(0, len(df) - base_window_size, stride):
                    end_idx = start_idx + base_window_size
                    window = df.iloc[start_idx:end_idx].copy()
                    mixed_windows.append(window)

        return mixed_windows