#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆评估组合赋权模型 - PCA 主成分分析

功能:
- 基于主成分分析 (PCA) 计算各评估维度的权重
- 采用"信息量"原则: 方差贡献率大的主成分对应更高的权重
- 计算最终综合得分

理论基础:
- PCA: 将高维数据投影到低维空间,保留最大方差
- 权重公式: W_j = Σ(λ_k · |u_{kj}|) / Σλ_k
  其中 λ_k 是第 k 个主成分的解释方差, u_{kj} 是第 j 个指标在第 k 个主成分上的载荷
- 解释: 指标在主要主成分上的载荷越大,该指标越重要
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict


class PCAWeightingModel:
    """
    基于 PCA 的组合赋权模型

    使用主成分分析计算各评估维度的客观权重，
    并计算最终的综合得分。
    """

    def __init__(self):
        """初始化模型"""
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.weights = None
        self.feature_names = None
        self.fitted = False

    def fit(self, df_profiles: pd.DataFrame) -> 'PCAWeightingModel':
        """
        训练 PCA 权重模型

        Parameters:
        -----------
        df_profiles : pd.DataFrame
            车辆画像表,必须包含以下 4 列:
            - Weibull_Score: 生命周期得分 (越高越新)
            - Usage_Score: 使用强度得分 (越低越激烈,但这里需要反转)
            - Maint_Score: 保养规范度得分 (越高越规范)
            - Reliability_Score: 可靠性得分 (越高越可靠)

        Returns:
        --------
        self : PCAWeightingModel
        """
        # 1. 提取特征列
        self.feature_names = ['Weibull_Score', 'Usage_Score', 'Maint_Score', 'Reliability_Score']

        # 检查列是否存在
        missing_cols = [col for col in self.feature_names if col not in df_profiles.columns]
        if missing_cols:
            raise ValueError(f"DataFrame 缺少必要的列: {missing_cols}")

        # 2. 提取数据并处理缺失值
        X = df_profiles[self.feature_names].copy()

        # 对于 Reliability_Score 的缺失值,用平均值填充
        if X['Reliability_Score'].isna().any():
            print(f"  ⚠ Reliability_Score 有 {X['Reliability_Score'].isna().sum()} 个缺失值,用均值填充")
            X['Reliability_Score'].fillna(X['Reliability_Score'].mean(), inplace=True)

        # 3. 反转 Usage_Score: 原始逻辑是"越低越激烈",但综合评分需要"越高越好"
        # 反转后: 100 - original_score,这样使用不激烈的车得分更高
        X['Usage_Score'] = 100.0 - X['Usage_Score']

        # 保存原始数据用于后续计算
        self.X_raw = X.values

        # 4. 标准化 (零均值,单位方差)
        print("\n【数据预处理】")
        X_scaled = self.scaler.fit_transform(X)
        print(f"  标准化完成: 均值≈0, 标准差≈1")

        # 5. PCA 拟合
        print("\n【PCA 主成分分析】")
        self.pca.fit(X_scaled)

        # 6. 计算权重
        print("\n【权重计算】")

        # 获取解释方差 (λ_k)
        explained_variance = self.pca.explained_variance_
        print(f"  各主成分解释方差: {explained_variance}")

        # 获取成分载荷矩阵 (u_{kj})
        # components_ 的形状是 (n_components, n_features)
        # 每一行是一个主成分,每一列是一个原始特征
        components = self.pca.components_
        print(f"  成分载荷矩阵形状: {components.shape}")

        # 计算权重: W_j = Σ(λ_k · |u_{kj}|) / Σλ_k
        n_features = len(self.feature_names)
        weights = np.zeros(n_features)

        for j in range(n_features):
            # 对第 j 个指标,计算其在所有主成分上的加权载荷和
            weighted_loadings = explained_variance * np.abs(components[:, j])
            weights[j] = weighted_loadings.sum()

        # 归一化权重,使其和为 1
        weights = weights / weights.sum()

        # 保存权重
        self.weights = dict(zip(self.feature_names, weights))

        # 7. 打印权重
        print("\n【各维度权重】")
        print("  " + "="*60)
        for feature, weight in self.weights.items():
            print(f"  {feature:20s}: {weight:.4f} ({weight*100:.2f}%)")
        print("  " + "="*60)
        print(f"  总计: {sum(self.weights.values()):.4f} (应等于 1.0000)")

        # 找出最重要的指标
        most_important = max(self.weights, key=self.weights.get)
        print(f"\n  💡 最重要指标: {most_important} (权重 {self.weights[most_important]:.2%})")

        # 8. 打印主成分解释方差比
        print("\n【主成分解释方差比】")
        for i, ratio in enumerate(self.pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
        cumulative = self.pca.explained_variance_ratio_.cumsum()
        print(f"  累计: {' '.join([f'{v:.4f}' for v in cumulative])}")

        self.fitted = True

        return self

    def calculate_score(self, df_profiles: pd.DataFrame) -> pd.DataFrame:
        """
        计算最终综合得分

        Parameters:
        -----------
        df_profiles : pd.DataFrame
            车辆画像表

        Returns:
        --------
        result_df : pd.DataFrame
            包含 Final_Score 列的 DataFrame
        """
        if not self.fitted:
            raise RuntimeError("模型尚未拟合,请先调用 fit() 方法")

        # 复制数据避免修改原表
        result_df = df_profiles.copy()

        # 处理缺失值
        X = result_df[self.feature_names].copy()
        if X['Reliability_Score'].isna().any():
            X['Reliability_Score'].fillna(X['Reliability_Score'].mean(), inplace=True)

        # 反转 Usage_Score (与 fit 时保持一致)
        X['Usage_Score'] = 100.0 - X['Usage_Score']

        # 计算加权得分
        final_scores = np.zeros(len(X))

        for feature, weight in self.weights.items():
            final_scores += X[feature] * weight

        # 添加到结果表
        result_df['Final_Score'] = final_scores

        return result_df

    def get_weights(self) -> Dict[str, float]:
        """
        获取计算出的权重

        Returns:
        --------
        weights : dict
            特征名到权重的映射
        """
        if not self.fitted:
            raise RuntimeError("模型尚未拟合,请先调用 fit() 方法")

        return self.weights.copy()
