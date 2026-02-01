#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆评估组合赋权模型 - PCA 主成分分析 + Transformer集成

功能:
- 基于主成分分析 (PCA) 计算各评估维度的权重
- 集成 Transformer 故障预测结果作为第五个维度
- 采用"信息量"原则: 方差贡献率大的主成分对应更高的权重
- 计算最终综合得分

理论基础:
- PCA: 将高维数据投影到低维空间,保留最大方差
- Transformer: 预测未来7/30/90天的故障风险
- 权重公式: W_j = Σ(λ_k · |u_{kj}|) / Σλ_k
  其中 λ_k 是第 k 个主成分的解释方差, u_{kj} 是第 j 个指标在第 k 个主成分上的载荷
- 解释: 指标在主要主成分上的载荷越大,该指标越重要
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, Optional, List
import os
import logging

# 导入Transformer预测服务
try:
    from data_loader import VehicleDataLoader
    from transformer_predictor import FailurePredictorService
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("  ⚠️  Transformer模块未找到，将使用4维评分（不包含故障预测）")


class PCAWeightingModel:
    """
    基于 PCA 的组合赋权模型（支持Transformer集成）

    使用主成分分析计算各评估维度的客观权重，
    可选集成Transformer故障预测结果作为第五个维度，
    并计算最终的综合得分。
    """

    def __init__(self, use_transformer: bool = True, model_path: Optional[str] = None):
        """
        初始化模型

        Parameters:
        -----------
        use_transformer : bool
            是否使用Transformer预测结果
        model_path : str, optional
            Transformer模型路径
        """
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.weights = None
        self.feature_names = None
        self.fitted = False
        self.use_transformer = use_transformer and TRANSFORMER_AVAILABLE
        self.transformer_service = None
        self.model_path = model_path

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if self.use_transformer and not model_path:
            # 默认模型路径
            default_model = "output/models/transformer_failure_predictor_enhanced_v2_best.pth"
            if os.path.exists(default_model):
                self.model_path = default_model
            else:
                self.logger.warning(f"默认模型 {default_model} 不存在，Transformer将被禁用")
                self.use_transformer = False

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
        # 1. 初始化Transformer预测服务（如果启用）
        if self.use_transformer:
            self._init_transformer_service()

        # 2. 提取基础特征列
        base_feature_names = ['Weibull_Score', 'Usage_Score', 'Maint_Score', 'Reliability_Score']

        # 检查基础列是否存在
        missing_cols = [col for col in base_feature_names if col not in df_profiles.columns]
        if missing_cols:
            raise ValueError(f"DataFrame 缺少必要的列: {missing_cols}")

        # 3. 处理基础数据
        X = df_profiles[base_feature_names].copy()

        # 对于 Reliability_Score 的缺失值,用平均值填充
        if X['Reliability_Score'].isna().any():
            print(f"  ⚠ Reliability_Score 有 {X['Reliability_Score'].isna().sum()} 个缺失值,用均值填充")
            X['Reliability_Score'].fillna(X['Reliability_Score'].mean(), inplace=True)

        # 反转 Usage_Score: 原始逻辑是"越低越激烈",但综合评分需要"越高越好"
        X['Usage_Score'] = 100.0 - X['Usage_Score']

        # 保存原始数据用于后续计算
        self.X_raw = X.values

        # 4. 如果使用Transformer，添加故障预测维度
        if self.use_transformer:
            transformer_scores = self._get_transformer_predictions(df_profiles)
            if transformer_scores is not None and len(transformer_scores) > 0:
                # 将Transformer预测结果添加到特征矩阵中
                # 注意：Transformer预测的是风险分数（越高风险越高），我们需要转换成可靠性分数
                reliability_score_from_transformer = 100.0 - np.array(transformer_scores)
                X['Transformer_Reliability_Score'] = reliability_score_from_transformer
                self.feature_names = ['Weibull_Score', 'Usage_Score', 'Maint_Score', 'Reliability_Score', 'Transformer_Reliability_Score']
                print(f"  ✓ 成功集成Transformer预测，使用5维评分系统")
            else:
                print(f"  ⚠ Transformer预测失败，回退到4维评分系统")
                self.feature_names = base_feature_names
                self.use_transformer = False
        else:
            self.feature_names = base_feature_names

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
        print("  " + "="*70)
        for feature, weight in self.weights.items():
            print(f"  {feature:25s}: {weight:.4f} ({weight*100:.2f}%)")
        print("  " + "="*70)
        print(f"  总计: {sum(self.weights.values()):.4f} (应等于 1.0000)")

        # 找出最重要的指标
        most_important = max(self.weights, key=self.weights.get)
        print(f"\n  💡 最重要指标: {most_important} (权重 {self.weights[most_important]:.2%})")

        # 显示使用的评分维度
        dimension_text = "5维" if self.use_transformer else "4维"
        print(f"\n  📊 当前评分维度: {dimension_text} {'(含Transformer预测)' if self.use_transformer else ''}")
        if self.use_transformer:
            print(f"     - 基础4维: 生命周期、使用强度、保养规范度、可靠性")
            print(f"     - 额外1维: Transformer故障预测（90天预测转换）")

        # 8. 打印主成分解释方差比
        print("\n【主成分解释方差比】")
        for i, ratio in enumerate(self.pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
        cumulative = self.pca.explained_variance_ratio_.cumsum()
        print(f"  累计: {' '.join([f'{v:.4f}' for v in cumulative])}")

        self.fitted = True

        return self

    def _init_transformer_service(self):
        """初始化Transformer预测服务"""
        try:
            if os.path.exists(self.model_path):
                self.transformer_service = FailurePredictorService(
                    model_path=self.model_path,
                    device='cpu'  # 默认使用CPU，避免GPU问题
                )
                print(f"  ✓ Transformer模型已加载: {self.model_path}")
            else:
                self.logger.warning(f"模型文件不存在: {self.model_path}")
                self.use_transformer = False
        except Exception as e:
            self.logger.error(f"初始化Transformer服务失败: {str(e)}")
            self.use_transformer = False

    def _get_transformer_predictions(self, df_profiles: pd.DataFrame) -> Optional[List[float]]:
        """
        获取Transformer预测结果

        Parameters:
        -----------
        df_profiles : pd.DataFrame
            车辆画像数据

        Returns:
        --------
        scores : List[float] or None
            转换后的可靠性得分列表，如果失败则返回None
        """
        if not self.transformer_service:
            return None

        try:
            scores = []
            # 对每个VIN进行预测
            for _, row in df_profiles.iterrows():
                vin = row['VIN']

                # 获取该VIN的时间线数据
                timeline_data = self._get_vehicle_timeline(vin)
                if timeline_data is not None and not timeline_data.empty:
                    # 获取故障预测（90天的风险）
                    predictions = self.transformer_service.predict_failure_risk(
                        timeline_data,
                        horizons=[90]  # 使用90天预测
                    )
                    # 将风险分数转换为可靠性分数
                    risk_score = predictions.get(90, 0.5)  # 默认0.5风险
                    reliability_score = 100.0 - risk_score * 100  # 转换为0-100的可靠性得分
                    scores.append(reliability_score)
                else:
                    # 如果没有时间线数据，使用默认分数
                    scores.append(50.0)  # 中性分数

            return scores

        except Exception as e:
            self.logger.error(f"获取Transformer预测失败: {str(e)}")
            return None

    def _get_vehicle_timeline(self, vin: str) -> Optional[pd.DataFrame]:
        """
        获取特定VIN的时间线数据

        Parameters:
        -----------
        vin : str
            车辆VIN码

        Returns:
        --------
        timeline_df : pd.DataFrame or None
            时间线数据，如果找不到则返回None
        """
        try:
            # 尝试从现有的数据文件中获取时间线数据
            if hasattr(self, '_timeline_cache'):
                timeline_df = self._timeline_cache
            else:
                # 假设vehicle_timeline.csv存在
                timeline_path = "data/vehicle_timeline.csv"
                if os.path.exists(timeline_path):
                    timeline_df = pd.read_csv(timeline_path)
                    self._timeline_cache = timeline_df
                else:
                    # 如果没有时间线文件，尝试从llm_parsed_results.csv构建
                    llm_path = "data/llm_parsed_results.csv"
                    if os.path.exists(llm_path):
                        llm_df = pd.read_csv(llm_path)
                        # 筛选特定VIN的数据
                        timeline_df = llm_df[llm_df['VIN'] == vin].copy()
                        # 如果需要，可以进一步处理
                        if not timeline_df.empty:
                            # 添加基本的时序信息
                            timeline_df['SETTLE_DATE'] = pd.to_datetime(timeline_df['SETTLE_DATE'])
                            timeline_df = timeline_df.sort_values('SETTLE_DATE')
                        self._timeline_cache = timeline_df
                    else:
                        return None

            # 返回特定VIN的数据
            if hasattr(self, '_timeline_cache'):
                vin_data = self._timeline_cache[self._timeline_cache['VIN'] == vin]
                if not vin_data.empty:
                    return vin_data

            return None

        except Exception as e:
            self.logger.error(f"获取车辆时间线数据失败: {str(e)}")
            return None

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

        # 处理Reliability_Score的缺失值
        if 'Reliability_Score' in X.columns and X['Reliability_Score'].isna().any():
            X['Reliability_Score'].fillna(X['Reliability_Score'].mean(), inplace=True)

        # 反转 Usage_Score (与 fit 时保持一致)
        if 'Usage_Score' in X.columns:
            X['Usage_Score'] = 100.0 - X['Usage_Score']

        # 如果使用Transformer，需要重新计算Transformer_Reliability_Score
        if self.use_transformer and 'Transformer_Reliability_Score' in self.feature_names:
            transformer_scores = self._get_transformer_predictions(df_profiles)
            if transformer_scores is not None and len(transformer_scores) == len(X):
                X['Transformer_Reliability_Score'] = transformer_scores
            else:
                # 如果无法获取新的预测，使用已有的分数
                pass

        # 计算加权得分
        final_scores = np.zeros(len(X))

        for feature, weight in self.weights.items():
            if feature in X.columns:
                final_scores += X[feature] * weight
            else:
                # 如果某个特征不存在，跳过
                print(f"  ⚠ 特征 {feature} 不存在，已跳过")

        # 添加到结果表
        result_df['Final_Score'] = final_scores

        # 添加评分维度说明
        dimension_text = "5维" if self.use_transformer else "4维"
        result_df['Scoring_Dimensions'] = dimension_text

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
