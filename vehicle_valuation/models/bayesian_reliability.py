#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贝叶斯更新可靠性模型 - 双模态可靠性评估

功能:
- 实现双模态可靠性评估 (Dual-Modal Reliability Assessment)
- 将Transformer的预测结果作为对"可靠性（Reliability）"维度的深度修正
- 支持94%普通车辆（仅历史数据）和6%富数据车辆（历史+预测数据）
- 实现不确定性惩罚机制，确保公平性

理论基础:
- History (历史得分): 由NHPP模型计算（基于历史故障密度）
- Future (未来得分): 由Transformer计算（基于序列模式预测）
- 动态融合函数: S_Final_Rel = w * S_History + (1-w) * S_Future
- 不确定性惩罚: λ_penalty = 0.95 (信息缺失惩罚分)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from .reliability import ReliabilityModel


class BayesianReliabilityModel:
    """
    贝叶斯更新可靠性评估模型

    采用双模态可靠性评估框架：
    - History: 历史得分（NHPP故障率模型）
    - Future: 未来得分（Transformer预测）
    - 动态融合权重: w = 0.6 (60%看历史，40%看未来)
    - 不确定性惩罚: λ_penalty = 0.95
    """

    def __init__(self, alpha: float = 0.6, penalty_factor: float = 0.95):
        """
        初始化贝叶斯可靠性模型

        Parameters:
        -----------
        alpha : float
            历史权重，默认0.6（60%看历史，40%看未来）
        penalty_factor : float
            不确定性惩罚因子，默认0.95
        """
        self.alpha = alpha  # 历史权重
        self.beta = 1.0 - alpha  # 未来权重
        self.penalty_factor = penalty_factor
        self.history_model = ReliabilityModel()
        self.transformer_risks = {}
        self.fitted = False

    def fit(self,
            llm_df: pd.DataFrame,
            base_df: pd.DataFrame,
            transformer_predictions: Optional[Dict[str, Dict]] = None):
        """
        训练双模态可靠性模型

        Parameters:
        -----------
        llm_df : pd.DataFrame
            LLM分析结果，必须包含ID, Severity列
        base_df : pd.DataFrame
            基础信息表，必须包含ID, VIN, REPAIR_MILEAGE列
        transformer_predictions : dict, optional
            Transformer预测结果字典 {vin: {'risk_90d': risk_value}}
        """
        # 1. 训练历史可靠性模型
        print("  → 步骤 1: 训练历史可靠性模型...")
        self.history_model.fit(llm_df, base_df)

        # 2. 保存Transformer预测结果
        if transformer_predictions:
            print(f"  → 步骤 2: 加载Transformer预测结果...")
            self.transformer_risks = transformer_predictions
            print(f"     成功加载 {len(self.transformer_risks)} 辆车的预测结果")
        else:
            print("  → 步骤 2: 无Transformer预测结果（普通车辆模式）")
            self.transformer_risks = {}

        # 计算统计信息
        self._calculate_statistics()

        self.fitted = True
        return self

    def calculate_hybrid_reliability(self, vin: str, nhpp_score: Optional[float] = None) -> float:
        """
        计算混合可靠性得分

        贝叶斯更新公式：
        - 有Transformer的车: S_Final = α * S_History + (1-α) * S_Future
        - 无Transformer的车: S_Final = S_History * λ_penalty

        其中: S_Future = (1 - P_risk) * 100

        Parameters:
        -----------
        vin : str
            车辆识别码
        nhpp_score : float, optional
            预计算的NHPP历史得分，如果为None则从模型获取

        Returns:
        --------
        final_score : float
            混合可靠性得分 (0-100)
        """
        if not self.fitted:
            raise RuntimeError("模型尚未拟合，请先调用fit()方法")

        # 获取历史得分
        if nhpp_score is None:
            nhpp_score = self.history_model.predict_score(vin)
            if nhpp_score is None:
                # 该车不在历史模型中，返回默认值
                return 50.0

        # 检查是否有Transformer预测
        if vin in self.transformer_risks:
            # 富数据车辆：使用贝叶斯更新
            risk_90d = self.transformer_risks[vin].get('risk_90d', 0.5)
            future_score = (1.0 - risk_90d) * 100

            # 贝叶斯更新：动态融合历史和未来
            final_score = (self.alpha * nhpp_score +
                          self.beta * future_score)

            scoring_type = "贝叶斯更新(5维)"
            confidence = "high"

        else:
            # 普通车辆：不确定性惩罚
            final_score = nhpp_score * self.penalty_factor
            scoring_type = "不确定性惩罚(4维)"
            confidence = "medium"

        # 确保得分在0-100范围内
        final_score = np.clip(final_score, 0.0, 100.0)

        return float(final_score)

    def get_reliability_breakdown(self, vin: str, nhpp_score: Optional[float] = None) -> Dict:
        """
        获取可靠性得分的详细分解

        Parameters:
        -----------
        vin : str
            车辆识别码
        nhpp_score : float, optional
            预计算的NHPP历史得分

        Returns:
        --------
        breakdown : dict
            包含详细分解信息的字典
        """
        if not self.fitted:
            raise RuntimeError("模型尚未拟合，请先调用fit()方法")

        # 获取历史得分
        if nhpp_score is None:
            nhpp_score = self.history_model.predict_score(vin)
            if nhpp_score is None:
                nhpp_score = 50.0
                has_history = False
            else:
                has_history = True
        else:
            has_history = True

        breakdown = {
            'vin': vin,
            'has_transformer': vin in self.transformer_risks,
            'has_history': has_history,
            'history_score': nhpp_score,
            'future_score': None,
            'final_score': None,
            'scoring_type': None,
            'weight_alpha': self.alpha,
            'weight_beta': self.beta,
            'penalty_factor': self.penalty_factor
        }

        if vin in self.transformer_risks:
            # 富数据车辆
            risk_90d = self.transformer_risks[vin].get('risk_90d', 0.5)
            future_score = (1.0 - risk_90d) * 100

            breakdown.update({
                'future_score': future_score,
                'final_score': self.alpha * nhpp_score + self.beta * future_score,
                'scoring_type': 'bayesian_update'
            })
        else:
            # 普通车辆
            breakdown.update({
                'final_score': nhpp_score * self.penalty_factor,
                'scoring_type': 'uncertainty_penalty'
            })

        # 确保数值范围正确
        for key in ['history_score', 'future_score', 'final_score']:
            if key in breakdown and breakdown[key] is not None:
                breakdown[key] = np.clip(breakdown[key], 0.0, 100.0)

        return breakdown

    def _calculate_statistics(self):
        """计算模型统计信息"""
        self.stats = {
            'total_vehicles': len(self.history_model.vehicle_profiles) if self.history_model.vehicle_profiles else 0,
            'transformer_vehicles': len(self.transformer_risks),
            'normal_vehicles': len(self.history_model.vehicle_profiles) - len(self.transformer_risks) if self.history_model.vehicle_profiles else 0,
            'transformer_coverage_rate': len(self.transformer_risks) / max(len(self.history_model.vehicle_profiles), 1)
        }

        # 分析Transformer预测的分布
        if self.transformer_risks:
            risks = [v.get('risk_90d', 0.5) for v in self.transformer_risks.values()]
            self.stats.update({
                'avg_risk_90d': np.mean(risks),
                'risk_90d_std': np.std(risks),
                'high_risk_vehicles': sum(1 for r in risks if r >= 0.6),
                'low_risk_vehicles': sum(1 for r in risks if r < 0.1)
            })

        print(f"  ✓ 统计信息计算完成")
        print(f"     总车辆数: {self.stats['total_vehicles']}")
        print(f"     有预测车辆: {self.stats['transformer_vehicles']} ({self.stats['transformer_coverage_rate']:.1%})")
        print(f"     无预测车辆: {self.stats['normal_vehicles']}")

        if self.transformer_risks:
            print(f"     Transformer预测平均风险: {self.stats['avg_risk_90d']:.1%}")
            print(f"     高风险车辆: {self.stats['high_risk_vehicles']} 辆")

    def get_model_summary(self) -> Dict:
        """
        获取模型摘要信息

        Returns:
        --------
        summary : dict
            模型摘要信息
        """
        if not self.fitted:
            raise RuntimeError("模型尚未拟合")

        return {
            'model_type': 'BayesianReliability',
            'alpha': self.alpha,
            'beta': self.beta,
            'penalty_factor': self.penalty_factor,
            'total_vehicles': self.stats['total_vehicles'],
            'transformer_coverage': self.stats['transformer_coverage_rate'],
            'fitted': self.fitted
        }