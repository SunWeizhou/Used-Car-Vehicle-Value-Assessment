#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆可靠性模型 - 故障率强度评估

功能:
- 基于维修记录严重程度评估车辆故障率强度
- 支持LLM标签和关键词规则两种识别方式
- 计算群体基准并进行指数归一化评分

理论基础:
- 故障率强度 Λ = Σ(权重) / 总里程
- 权重体系: L0=0, L1=1, L2=5, L3=20
- 评分公式: Score = 100 × exp(-0.693 × Λ_i / Λ_pop)
  - Λ_i = Λ_pop 时，得分 = 50 分
  - Λ_i = 0 时，得分 = 100 分
  - Λ_i = 2×Λ_pop 时，得分 = 25 分
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


class ReliabilityModel:
    """
    车辆可靠性评估模型

    基于维修记录的严重程度计算故障率强度，
    并通过与群体基准的对比进行评分。
    """

    def __init__(self):
        """初始化模型"""
        # 严重程度权重映射
        self.weights = {'L0': 0, 'L1': 1, 'L2': 5, 'L3': 20}

        self.baseline_lambda = None  # 群体基准故障率强度
        self.vehicle_lambdas = None  # 每辆车的故障率强度
        self.fitted = False

    def _get_severity(self, row: pd.Series) -> str:
        """
        获取维修记录的严重程度等级

        Parameters:
        -----------
        row : pd.Series
            单条维修记录，需包含 FAULT_DESC 和可能的 Severity 列

        Returns:
        --------
        severity : str
            严重程度等级 ('L0', 'L1', 'L2', 'L3')
        """
        # 优先使用 LLM 生成的 Severity 标签
        if 'Severity' in row and pd.notna(row['Severity']) and row['Severity'] in self.weights:
            return row['Severity']

        # 兜底规则：基于关键词匹配
        fault_desc = str(row['FAULT_DESC']).lower()

        # L0: 保养类
        maintenance_keywords = ['保养', '机油', '三滤', '润滑油', '机滤']
        if any(keyword in fault_desc for keyword in maintenance_keywords):
            return 'L0'

        # L3: 重大故障 (大修、事故、核心部件)
        major_keywords = ['大修', '事故', '发动机', '变速箱', '大梁', '制动', '转向']
        # 排除保养相关
        if any(keyword in fault_desc for keyword in major_keywords):
            if '保养' not in fault_desc:
                return 'L3'

        # L1: 其他轻微故障 (默认)
        return 'L1'

    def fit(self, df_clean: pd.DataFrame, df_llm: Optional[pd.DataFrame] = None):
        """
        训练可靠性模型

        计算每辆车的故障率强度：
        Λ_i = Σ(权重) / 总里程

        Parameters:
        -----------
        df_clean : pd.DataFrame
            基础信息表 (清洗后)，需包含 VIN, REPAIR_MILEAGE, FAULT_DESC, ID
        df_llm : pd.DataFrame, optional
            LLM 结构化结果表，包含 ID, Severity 列
        """
        # 1. 数据准备
        df = df_clean.copy()

        # 2. 合并 LLM 结果（如果有）
        if df_llm is not None:
            df = df.merge(
                df_llm[['ID', 'Severity']],
                on='ID',
                how='left'
            )

        # 3. 计算每条记录的权重
        df['severity'] = df.apply(self._get_severity, axis=1)
        df['weight'] = df['severity'].map(self.weights)

        # 4. 按 VIN 聚合计算故障率强度
        vehicle_stats = df.groupby('VIN').agg({
            'REPAIR_MILEAGE': 'max',  # 最大里程
            'weight': 'sum'            # 权重总和
        }).reset_index()

        vehicle_stats.rename(columns={
            'REPAIR_MILEAGE': 'max_mileage',
            'weight': 'total_weight'
        }, inplace=True)

        # 5. 计算故障率强度 Λ = Σ权重 / 里程
        # 避免除零，设置最小里程为 1000 km
        vehicle_stats['max_mileage'] = vehicle_stats['max_mileage'].clip(lower=1000)
        vehicle_stats['lambda'] = vehicle_stats['total_weight'] / vehicle_stats['max_mileage']

        # 6. 计算群体基准 (所有车辆 Λ 的平均值)
        self.baseline_lambda = vehicle_stats['lambda'].mean()

        # 7. 保存每辆车的故障率强度
        self.vehicle_lambdas = vehicle_stats.set_index('VIN')['lambda'].to_dict()

        # 8. 保存统计信息
        self.stats = {
            'n_vehicles': len(vehicle_stats),
            'lambda_mean': vehicle_stats['lambda'].mean(),
            'lambda_median': vehicle_stats['lambda'].median(),
            'lambda_min': vehicle_stats['lambda'].min(),
            'lambda_max': vehicle_stats['lambda'].max(),
            'total_weight_mean': vehicle_stats['total_weight'].mean()
        }

        self.fitted = True

        return self

    def predict_score(self, vin: str, current_mileage: float = None, fault_history: list = None) -> float:
        """
        预测可靠性得分

        Score = 100 × exp(-0.693 × Λ_i / Λ_pop)

        解释:
        - Λ_i = Λ_pop 时，得分 = 50 分
        - Λ_i = 0 时，得分 = 100 分
        - Λ_i = 2×Λ_pop 时，得分 = 25 分

        Parameters:
        -----------
        vin : str
            车辆识别码
        current_mileage : float, optional
            当前里程 (简化版工程实现中不使用，直接查表)
        fault_history : list, optional
            故障历史 (简化版工程实现中不使用，直接查表)

        Returns:
        --------
        score : float
            可靠性得分 (0-100)
        """
        if not self.fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        # 查找该 VIN 的故障率强度
        lambda_i = self.vehicle_lambdas.get(vin)

        # 如果 VIN 不在训练集中，返回默认分
        if lambda_i is None:
            return 80.0

        # 计算得分: 100 × exp(-0.693 × Λ_i / Λ_pop)
        score = 100.0 * np.exp(-0.693 * lambda_i / self.baseline_lambda)

        return float(score)
