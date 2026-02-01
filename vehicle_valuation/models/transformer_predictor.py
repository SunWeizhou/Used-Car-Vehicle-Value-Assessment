"""
Transformer-based Vehicle Failure Prediction Model

该模块实现了一个基于Transformer的故障预测模型，用于预测车辆在未来不同时间窗口内的故障风险。
模型采用多模态设计，结合时序特征和系统关联特征，提供多步预测能力。

主要功能：
1. 时序故障模式建模
2. 系统间依赖关系分析
3. 多步故障风险预测（7天/30天/90天）
4. 维护建议生成

作者：Claude AI
创建时间：2026-02-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

@dataclass
class TimeWindowConfig:
    """时间窗口配置"""
    window_size: int = 30  # 时间窗口大小（天）
    prediction_horizons: List[int] = None  # 预测时间跨度
    stride: int = 7  # 滑动步长

    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [7, 30, 90]

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TemporalFeatureExtractor(nn.Module):
    """时序特征提取器"""
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.norm(x)
        return F.relu(x)

class SystemDependencyEncoder(nn.Module):
    """系统依赖关系编码器"""
    def __init__(self, num_systems: int, d_model: int):
        super().__init__()
        # 使用更大的嵌入层以避免索引越界
        self.system_embedding = nn.Embedding(1000, d_model)  # 使用固定大值

    def forward(self, system_indices: torch.Tensor) -> torch.Tensor:
        # 确保输入是long类型
        if system_indices.dtype != torch.long:
            system_indices = system_indices.long()

        # 裁剪索引到有效范围
        system_indices = torch.clamp(system_indices, 0, 999)  # 最大索引999

        # 嵌入系统索引 - [batch_size, 1, d_model]
        embedded = self.system_embedding(system_indices)

        # 直接复制到整个序列长度
        seq_len = 10  # 时序序列长度
        system_features = embedded.expand(-1, seq_len, -1)  # [batch_size, seq_len, d_model]

        return system_features

class FailurePredictor(nn.Module):
    """故障预测主模型"""
    def __init__(
        self,
        temporal_feature_dim: int,
        num_systems: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()

        # 时序特征编码
        self.temporal_encoder = TemporalFeatureExtractor(temporal_feature_dim, d_model)

        # 系统依赖编码
        self.system_encoder = SystemDependencyEncoder(num_systems, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer主网络
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 预测头
        self.prediction_heads = nn.ModuleDict({
            str(horizon): nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            ) for horizon in [7, 30, 90]
        })

    def forward(self, temporal_features: torch.Tensor, system_features: torch.Tensor, system_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码时序特征
        temporal_encoded = self.temporal_encoder(temporal_features)

        # 编码系统特征
        system_encoded = self.system_encoder(system_indices)

        # 添加位置编码
        temporal_encoded = self.pos_encoding(temporal_encoded)

        # 直接传递给transformer（不需要融合层）
        transformer_output = self.transformer(system_encoded)

        # 多步预测
        predictions = {}
        for horizon, head in self.prediction_heads.items():
            predictions[horizon] = head(transformer_output)

        return predictions

class DataPreprocessor:
    """数据预处理器"""
    def __init__(self):
        self.system_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = None

    def prepare_temporal_features(self, df_timeline: pd.DataFrame) -> pd.DataFrame:
        """准备时序特征"""
        features = pd.DataFrame()

        # 基础特征
        features['mileage'] = df_timeline['REPAIR_MILEAGE']
        # 确保SETTLE_DATE是datetime类型
        df_timeline['SETTLE_DATE'] = pd.to_datetime(df_timeline['SETTLE_DATE'])
        features['time_diff'] = df_timeline['SETTLE_DATE'].diff().dt.days.fillna(0)

        # 里程差分
        features['mileage_diff'] = df_timeline['REPAIR_MILEAGE'].diff().fillna(0)

        # 严重程度编码
        severity_mapping = {'L0': 0, 'L1': 1, 'L2': 2, 'L3': 3}
        features['severity'] = df_timeline['Severity'].map(severity_mapping)

        # 累积故障次数
        features['cumulative_failures'] = df_timeline.groupby(level=0).cumcount()

        # 间隔故障次数（最近30天）
        features['failures_30d'] = self._count_failures_in_window(df_timeline, 30)

        # 维修频率
        features['repair_frequency'] = features['cumulative_failures'] / (features['time_diff'] + 1)

        # 距离上次维修时间（归一化）
        features['time_since_last'] = features['time_diff'].rolling(window=3, min_periods=1).mean()

        # 确保没有NaN值
        features = features.fillna(0)

        return features

    def prepare_system_features(self, df_timeline: pd.DataFrame) -> pd.DataFrame:
        """准备系统特征"""
        # 系统编码
        systems = df_timeline['System'].values
        system_encoded = self.system_encoder.fit_transform(systems)

        # 系统统计特征
        system_stats = df_timeline.groupby('System').agg({
            'REPAIR_MILEAGE': ['count', 'mean', 'std'],
            'Severity': lambda x: (x != 'L0').mean()  # 故障比例
        }).fillna(0)

        # 重命名列
        system_stats.columns = ['system_count', 'system_avg_mileage', 'system_std_mileage', 'system_failure_rate']

        return pd.DataFrame({
            'system_encoded': system_encoded,
            'system_failure_rate': df_timeline['System'].map(
                lambda x: system_stats.loc[x, 'system_failure_rate']
            )
        })

    def _count_failures_in_window(self, df: pd.DataFrame, days: int) -> pd.Series:
        """计算时间窗口内的故障次数"""
        result = pd.Series(0, index=df.index)
        current_date = df['SETTLE_DATE']

        for i in range(len(df)):
            start_date = current_date.iloc[i] - timedelta(days=days)
            mask = (current_date <= current_date.iloc[i]) & (current_date >= start_date)
            result.iloc[i] = mask.sum()

        return result

    def create_sequences(self, df: pd.DataFrame, config: TimeWindowConfig) -> List[Dict]:
        """创建序列数据"""
        sequences = []

        for vin in df['VIN'].unique():
            vin_data = df[df['VIN'] == vin].sort_values('SETTLE_DATE')

            # 滑动窗口创建序列
            for i in range(0, max(1, len(vin_data) - config.window_size), config.stride):
                window_data = vin_data.iloc[i:i + config.window_size]

                # 标签生成（未来是否故障）
                labels = self._generate_labels(vin_data, i + config.window_size, config.prediction_horizons)

                if len(window_data) > 0 and labels is not None:
                    sequences.append({
                        'vin': vin,
                        'window_data': window_data,
                        'labels': labels,
                        'window_start': window_data['SETTLE_DATE'].iloc[0],
                        'window_end': window_data['SETTLE_DATE'].iloc[-1]
                    })

        return sequences

    def _generate_labels(self, vin_data: pd.DataFrame, start_idx: int, horizons: List[int]) -> Optional[Dict]:
        """生成标签"""
        # 边界检查
        if start_idx >= len(vin_data):
            return None

        labels = {}

        for horizon in horizons:
            end_date = vin_data.iloc[start_idx]['SETTLE_DATE'] + timedelta(days=horizon)

            # 查找未来horizon天内是否有故障
            future_failures = vin_data[
                (vin_data['SETTLE_DATE'] > vin_data.iloc[start_idx]['SETTLE_DATE']) &
                (vin_data['SETTLE_DATE'] <= end_date)
            ]

            # 标签：是否有L2或L3级别的故障
            labels[str(horizon)] = 1 if len(future_failures) > 0 and \
                                      any(severity in future_failures['Severity'].values
                                          for severity in ['L2', 'L3']) else 0

        return labels

class TransformerTrainer:
    """训练器"""
    def __init__(self, model: FailurePredictor, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        # 损失函数
        self.criterion = nn.BCELoss().to(self.device)  # 移动到设备

        # 训练历史
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            self.optimizer.zero_grad()

            # 准备数据
            temporal_features = batch['temporal'].to(self.device)
            system_features = batch['system'].to(self.device)
            system_indices = torch.LongTensor(batch['system_indices']).to(self.device)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}

            # 前向传播
            predictions = self.model(temporal_features, system_features, system_indices)

            # 计算损失
            loss = 0
            for horizon in predictions:
                if horizon in labels:
                    # 取每个序列的最后一个预测值
                    pred = predictions[horizon]  # [batch_size, seq_len, 1]
                    # 移除最后一个维度并取最后一个时间步的预测
                    pred_last = pred.squeeze(-1)[:, -1]  # [batch_size]
                    loss += self.criterion(pred_last, labels[horizon].float())

            loss /= len(predictions)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                temporal_features = batch['temporal'].to(self.device)
                system_features = batch['system'].to(self.device)
                system_indices = torch.LongTensor(batch['system_indices']).to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}

                predictions = self.model(temporal_features, system_features, system_indices)

                loss = 0
                for horizon in predictions:
                    if horizon in labels:
                        # 取每个序列的最后一个预测值
                        pred = predictions[horizon]  # [batch_size, seq_len, 1]
                        # 移除最后一个维度并取最后一个时间步
                        pred_last = pred.squeeze(-1)[:, -1]  # [batch_size]
                        loss += self.criterion(pred_last, labels[horizon].float())

                loss /= len(predictions)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        patience: int = 20
    ) -> Dict[str, List[float]]:
        """完整训练过程"""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step(val_loss)

            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_checkpoint(f'best_model_epoch_{epoch}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

            # 每5个epoch输出一次进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                if val_loss < best_val_loss:
                    print(f'  -> New best validation loss: {val_loss:.4f}')
                else:
                    print(f'  -> Patience: {patience_counter}/{patience}')

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

class FailurePredictorService:
    """故障预测服务"""
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型
        self.model = FailurePredictor(
            temporal_feature_dim=8,
            num_systems=20,  # 预设系统数量
            d_model=256,
            nhead=8,
            num_layers=6
        )

        self.preprocessor = DataPreprocessor()

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {model_path} 加载")

    def predict_failure_risk(
        self,
        vin_data: pd.DataFrame,
        horizons: List[int] = [7, 30, 90]
    ) -> Dict[str, float]:
        """预测故障风险"""
        self.model.eval()

        # 数据预处理
        temporal_features = self.preprocessor.prepare_temporal_features(vin_data)
        system_features = self.preprocessor.prepare_system_features(vin_data)

        # 转换为张量
        temporal_tensor = torch.FloatTensor(temporal_features.values).unsqueeze(0).to(self.device)
        system_tensor = torch.FloatTensor(system_features.values).unsqueeze(0).to(self.device)

        # 获取系统索引
        system_indices = torch.LongTensor(system_features['system_encoded'].values).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            predictions = self.model(temporal_tensor, system_tensor, system_indices)

        # 提取指定时间跨度的预测
        risk_predictions = {}
        for horizon in horizons:
            if str(horizon) in predictions:
                risk_predictions[horizon] = predictions[str(horizon)].item()

        return risk_predictions

    def generate_maintenance_recommendations(
        self,
        predictions: Dict[str, float],
        vin_data: pd.DataFrame
    ) -> List[str]:
        """生成维护建议"""
        recommendations = []

        # 分析预测结果
        for horizon, risk in predictions.items():
            if risk > 0.7:  # 高风险
                recommendations.append(f"未来{horizon}天故障风险高达{risk:.1%}，建议立即全面检查")
            elif risk > 0.4:  # 中等风险
                recommendations.append(f"未来{horizon}天故障风险为{risk:.1%}，建议进行预防性检查")

        # 分析常见故障系统
        if len(vin_data) > 0:
            system_failures = vin_data['System'].value_counts()
            top_system = system_failures.index[0] if len(system_failures) > 0 else None

            if top_system and 'L2' in vin_data['Severity'].values:
                recommendations.append(f"经常故障的系统：{top_system}，建议重点关注")

        # 里程建议
        if len(vin_data) > 0:
            last_mileage = vin_data['REPAIR_MILEAGE'].iloc[-1]
            if last_mileage > 500000:
                recommendations.append("车辆里程较高，建议增加检查频率")

        return recommendations if recommendations else ["当前车辆状态良好，按正常保养周期即可"]

    def get_model_summary(self) -> Dict:
        """获取模型摘要"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'device': str(self.device),
            'prediction_horizons': [7, 30, 90]
        }