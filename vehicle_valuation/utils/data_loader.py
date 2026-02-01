"""
数据加载器模块

为Transformer故障预测模型提供数据加载和处理功能。

主要功能：
1. 时序数据加载和预处理
2. 批次数据生成
3. 数据验证
4. 数据集划分

作者：Claude AI
创建时间：2026-02-01
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta

class VehicleFailureDataset(Dataset):
    """车辆故障数据集"""
    def __init__(
        self,
        sequences: List[Dict],
        temporal_feature_columns: List[str],
        system_feature_columns: List[str],
        prediction_horizons: List[int] = [7, 30, 90]
    ):
        self.sequences = sequences
        self.temporal_feature_columns = temporal_feature_columns
        self.system_feature_columns = system_feature_columns
        self.prediction_horizons = prediction_horizons

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        sequence = self.sequences[idx]

        # 调试信息
        if idx < 3:  # 只打印前3个序列的调试信息
            print(f"Debug: sequence {idx} keys: {list(sequence.keys())}")
            if 'window_data' in sequence:
                print(f"Debug: window_data shape: {sequence['window_data'].shape}")
                print(f"Debug: window_data columns: {list(sequence['window_data'].columns)}")
            else:
                print(f"Debug: No 'window_data' key in sequence {idx}")

        # 安全获取时序特征
        if 'window_data' not in sequence:
            raise KeyError(f"序列 {idx} 缺少 'window_data' 键")

        window_data = sequence['window_data']

        # 检查必需的列是否存在
        missing_temporal = [col for col in self.temporal_feature_columns if col not in window_data.columns]
        if missing_temporal:
            print(f"警告：时序特征缺失列: {missing_temporal}")
            # 使用存在的列
            available_cols = [col for col in self.temporal_feature_columns if col in window_data.columns]
            temporal_features = window_data[available_cols].values
        else:
            temporal_features = window_data[self.temporal_feature_columns].values

        # 确保时序特征是正确的形状
        if temporal_features.shape[1] != len(self.temporal_feature_columns):
            # 如果不匹配，使用前n个特征
            temporal_features = temporal_features[:, :len(self.temporal_feature_columns)]

        # 安全获取系统特征
        missing_system = [col for col in self.system_feature_columns if col not in window_data.columns]
        if missing_system:
            print(f"警告：系统特征缺失列: {missing_system}")
            # 使用存在的列
            available_cols = [col for col in self.system_feature_columns if col in window_data.columns]
            system_features = window_data[available_cols].values
        else:
            system_features = window_data[self.system_feature_columns].values

        # 获取系统索引（取第一个时间点的系统）
        system_encoded = system_features[0, 0] if len(system_features) > 0 else 0

        # 准备标签
        labels = {}
        for horizon in self.prediction_horizons:
            labels[str(horizon)] = sequence['labels'][str(horizon)]

        return {
            'temporal': torch.FloatTensor(temporal_features),
            'system': torch.FloatTensor(system_features),
            'system_indices': torch.LongTensor([system_encoded]),
            'labels': labels,
            'vin': sequence['vin'],
            'window_start': sequence['window_start'],
            'window_end': sequence['window_end']
        }

class VehicleDataLoader:
    """数据加载器"""
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessor = None
        self.temporal_scaler = None
        self.system_scaler = None

    def load_data(self, data_path: str) -> pd.DataFrame:
        """加载数据"""
        try:
            df = pd.read_csv(data_path)
            print(f"成功加载数据：{len(df)}条记录")
            return df
        except Exception as e:
            print(f"数据加载失败：{e}")
            return None

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """预处理数据"""
        if self.preprocessor is None:
            from models.transformer_predictor import DataPreprocessor
            self.preprocessor = DataPreprocessor()

        # 准备特征
        temporal_features = self.preprocessor.prepare_temporal_features(df)
        system_features = self.preprocessor.prepare_system_features(df)

        # 数据标准化
        if self.temporal_scaler is None:
            self.temporal_scaler = StandardScaler()
            self.system_scaler = StandardScaler()

        temporal_features = pd.DataFrame(
            self.temporal_scaler.fit_transform(temporal_features),
            columns=temporal_features.columns,
            index=temporal_features.index
        )

        system_features = pd.DataFrame(
            self.system_scaler.fit_transform(system_features),
            columns=system_features.columns,
            index=system_features.index
        )

        return temporal_features, system_features

    def create_sequences(self, df: pd.DataFrame, config: Dict) -> List[Dict]:
        """创建序列"""
        from models.transformer_predictor import TimeWindowConfig

        time_config = TimeWindowConfig(
            window_size=config.get('window_size', 30),
            prediction_horizons=config.get('prediction_horizons', [7, 30, 90]),
            stride=config.get('stride', 7)
        )

        # 创建原始序列
        raw_sequences = self.preprocessor.create_sequences(df, time_config)

        # 确保已经预处理
        if self.temporal_scaler is None or self.system_scaler is None:
            self.preprocess_data(df)

        # 为每个序列添加预处理的特征
        sequences_with_features = []
        for seq in raw_sequences:
            # 直接使用window_data进行特征提取
            window_data = seq['window_data']

            # 预处理该窗口的数据
            temporal_data = self.preprocessor.prepare_temporal_features(window_data)
            system_data = self.preprocessor.prepare_system_features(window_data)

            # 标准化
            temporal_scaled = pd.DataFrame(
                self.temporal_scaler.transform(temporal_data),
                columns=temporal_data.columns,
                index=temporal_data.index
            )

            system_scaled = pd.DataFrame(
                self.system_scaler.transform(system_data),
                columns=system_data.columns,
                index=system_data.index
            )

            # 合并特征
            all_features = pd.concat([temporal_scaled, system_scaled], axis=1)

            # 创建新的序列结构
            new_seq = {
                'vin': seq['vin'],
                'window_start': seq['window_start'],
                'window_end': seq['window_end'],
                'window_data': all_features,  # 使用预处理后的特征
                'labels': seq['labels']
            }

            sequences_with_features.append(new_seq)

        return sequences_with_features

    def create_datasets(self, sequences: List[Dict]) -> Tuple[Dataset, Dataset]:
        """创建训练和验证数据集"""
        # 定义特征列
        temporal_columns = [
            'mileage', 'time_diff', 'mileage_diff', 'severity',
            'cumulative_failures', 'failures_30d', 'repair_frequency', 'time_since_last'
        ]

        system_columns = [
            'system_encoded', 'system_failure_rate'
        ]

        # 划分数据
        train_sequences, val_sequences = train_test_split(
            sequences,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        # 创建数据集
        train_dataset = VehicleFailureDataset(
            train_sequences,
            temporal_columns,
            system_columns,
            [7, 30, 90]
        )

        val_dataset = VehicleFailureDataset(
            val_sequences,
            temporal_columns,
            system_columns,
            [7, 30, 90]
        )

        print(f"训练集大小：{len(train_dataset)}")
        print(f"验证集大小：{len(val_dataset)}")

        return train_dataset, val_dataset

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

        return train_loader, val_loader

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """批次数据整理"""
        temporal_batch = torch.stack([item['temporal'] for item in batch])
        system_batch = torch.stack([item['system'] for item in batch])
        system_indices_batch = torch.stack([item['system_indices'] for item in batch])

        # 处理标签
        labels_batch = {}
        for item in batch:
            for key, value in item['labels'].items():
                if key not in labels_batch:
                    labels_batch[key] = []
                labels_batch[key].append(value)

        # 转换标签为张量
        for key in labels_batch:
            labels_batch[key] = torch.FloatTensor(labels_batch[key])

        return {
            'temporal': temporal_batch,
            'system': system_batch,
            'system_indices': system_indices_batch,
            'labels': labels_batch,
            'vins': [item['vin'] for item in batch],
            'window_starts': [item['window_start'] for item in batch],
            'window_ends': [item['window_end'] for item in batch]
        }

class DataValidator:
    """数据验证器"""
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict:
        """验证数据质量"""
        validation_report = {
            'total_records': len(df),
            'missing_values': {},
            'outliers': {},
            'duplicates': {},
            'date_range': {},
            'quality_score': 0
        }

        # 缺失值检查
        for column in df.columns:
            missing_count = df[column].isna().sum()
            validation_report['missing_values'][column] = {
                'count': missing_count,
                'percentage': missing_count / len(df) * 100
            }

        # 异常值检查
        if 'REPAIR_MILEAGE' in df.columns:
            q1 = df['REPAIR_MILEAGE'].quantile(0.25)
            q3 = df['REPAIR_MILEAGE'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df['REPAIR_MILEAGE'] < lower_bound) |
                         (df['REPAIR_MILEAGE'] > upper_bound)]
            validation_report['outliers']['REPAIR_MILEAGE'] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(df) * 100
            }

        # 重复记录检查
        duplicates = df.duplicated()
        validation_report['duplicates']['total'] = {
            'count': duplicates.sum(),
            'percentage': duplicates.sum() / len(df) * 100
        }

        # 日期范围检查
        if 'SETTLE_DATE' in df.columns:
            df['SETTLE_DATE'] = pd.to_datetime(df['SETTLE_DATE'])
            validation_report['date_range'] = {
                'start': df['SETTLE_DATE'].min(),
                'end': df['SETTLE_DATE'].max(),
                'span_days': (df['SETTLE_DATE'].max() - df['SETTLE_DATE'].min()).days
            }

        # 计算质量分数
        quality_score = 100
        # 缺失值扣分
        for col in validation_report['missing_values']:
            if validation_report['missing_values'][col]['percentage'] > 10:
                quality_score -= 10
            elif validation_report['missing_values'][col]['percentage'] > 5:
                quality_score -= 5

        # 异常值扣分
        if 'REPAIR_MILEAGE' in validation_report['outliers']:
            if validation_report['outliers']['REPAIR_MILEAGE']['percentage'] > 5:
                quality_score -= 5

        # 重复记录扣分
        if validation_report['duplicates']['total']['percentage'] > 1:
            quality_score -= 5

        validation_report['quality_score'] = max(0, quality_score)

        return validation_report

    @staticmethod
    def validate_sequences(sequences: List[Dict]) -> Dict:
        """验证序列质量"""
        validation_report = {
            'total_sequences': len(sequences),
            'sequence_lengths': {},
            'label_distribution': {},
            'vin_distribution': {},
            'quality_issues': []
        }

        # 序列长度分析
        lengths = [len(seq['window_data']) for seq in sequences]
        validation_report['sequence_lengths'] = {
            'min': min(lengths),
            'max': max(lengths),
            'mean': np.mean(lengths),
            'std': np.std(lengths)
        }

        # 标签分布分析
        for horizon in ['7', '30', '90']:
            labels = [seq['labels'][horizon] for seq in sequences]
            validation_report['label_distribution'][horizon] = {
                'positive': sum(labels),
                'negative': len(labels) - sum(labels),
                'positive_ratio': sum(labels) / len(labels)
            }

        # VIN分布分析
        vin_counts = {}
        for seq in sequences:
            vin = seq['vin']
            vin_counts[vin] = vin_counts.get(vin, 0) + 1

        validation_report['vin_distribution'] = {
            'unique_vins': len(vin_counts),
            'avg_sequences_per_vin': np.mean(list(vin_counts.values())),
            'max_sequences_per_vin': max(vin_counts.values()),
            'min_sequences_per_vin': min(vin_counts.values())
        }

        # 质量问题检查
        if validation_report['label_distribution']['7']['positive_ratio'] < 0.01:
            validation_report['quality_issues'].append("7天预测标签正样本过少")

        if validation_report['label_distribution']['90']['positive_ratio'] > 0.5:
            validation_report['quality_issues'].append("90天预测标签正样本过多")

        if validation_report['sequence_lengths']['max'] < 10:
            validation_report['quality_issues'].append("序列长度过短")

        return validation_report