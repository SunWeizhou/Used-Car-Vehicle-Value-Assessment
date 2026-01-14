#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆使用行为模型 - ECDF 评估

功能:
- 使用经验累积分布函数 (ECDF) 评估车辆使用强度
- 评估保养规范度
- 计算行为得分（使用强度、保养规范度）

理论基础:
- ECDF (Empirical Cumulative Distribution Function): F_n(x) = (1/n) * Σ[I(x_i ≤ x)]
- 使用强度得分: 100 * (1 - F(日里程)) - 跑得越凶，得分越低
- 保养规范度得分: 100 * F(保养密度) - 保养越勤，得分越高
"""

import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from pathlib import Path
from typing import Tuple, Dict, Optional


class BehaviorModel:
    """
    车辆使用行为评估模型

    基于 ECDF 评估车辆的使用强度和保养规范度。
    """

    def __init__(self):
        """初始化模型"""
        self.ecdf_usage = None  # 使用强度的 ECDF
        self.ecdf_maint = None  # 保养规范度的 ECDF
        self.fitted = False

    def fit(self, df_base: pd.DataFrame, df_llm: Optional[pd.DataFrame] = None):
        """
        训练行为模型

        构建指标：
        1. avg_daily_mileage: 平均日里程 = 总里程 / 使用天数
        2. maint_density: 保养密度 = 保养次数 / 总里程 * 10000

        Parameters:
        -----------
        df_base : pd.DataFrame
            基础信息表 (清洗后)，需包含 VIN, REPAIR_MILEAGE, SETTLE_DATE, FAULT_DESC
        df_llm : pd.DataFrame, optional
            LLM 结构化结果表，包含 Event_Type 列用于识别保养
        """
        # 1. 数据准备
        df_base = df_base.copy()
        df_base['SETTLE_DATE'] = pd.to_datetime(df_base['SETTLE_DATE'])

        # 2. 如果没有 LLM 结果，使用简单规则识别保养
        if df_llm is None:
            # 简单规则：故障描述包含保养关键词
            maintenance_keywords = ['保养', '更换机油', '机滤', '三滤', '润滑油']
            df_base['is_maintenance'] = df_base['FAULT_DESC'].str.contains(
                '|'.join(maintenance_keywords), na=False
            )
        else:
            # 使用 LLM 结果：将 Event_Type 映射回原数据
            # 假设 df_llm 有 ID 列可以与 df_base 关联
            df_base = df_base.merge(
                df_llm[['ID', 'Event_Type']],
                on='ID',
                how='left'
            )
            df_base['is_maintenance'] = (df_base['Event_Type'] == '保养').fillna(False)

        # 3. 按 VIN 聚合计算指标
        vehicle_stats = df_base.groupby('VIN').agg({
            'REPAIR_MILEAGE': 'max',  # 最大里程
            'SETTLE_DATE': ['min', 'max'],  # 首次和最后出现日期
            'is_maintenance': 'sum'  # 保养次数
        }).reset_index()

        # 展平多级列名
        vehicle_stats.columns = ['VIN', 'max_mileage', 'first_date', 'last_date', 'maint_count']

        # 4. 计算使用天数 (span_days)
        vehicle_stats['span_days'] = (vehicle_stats['last_date'] - vehicle_stats['first_date']).dt.days

        # 过滤异常值：span_days 太小 (< 30天) 的车辆设为 30 天
        vehicle_stats.loc[vehicle_stats['span_days'] < 30, 'span_days'] = 30

        # 5. 计算指标
        # avg_daily_mileage = 总里程 / 使用天数
        vehicle_stats['avg_daily_mileage'] = vehicle_stats['max_mileage'] / vehicle_stats['span_days']

        # maint_density = 保养次数 / 总里程 * 10000 (每万公里保养次数)
        vehicle_stats['maint_density'] = vehicle_stats['maint_count'] / vehicle_stats['max_mileage'] * 10000

        # 6. 过滤无效数据
        # 移除 avg_daily_mileage 或 maint_density 为 NaN 或 Inf 的行
        vehicle_stats = vehicle_stats[
            (vehicle_stats['avg_daily_mileage'] > 0) &
            (vehicle_stats['avg_daily_mileage'] < np.inf) &
            (vehicle_stats['maint_density'] >= 0) &
            (vehicle_stats['maint_density'] < np.inf)
        ]

        # 7. 拟合 ECDF
        self.ecdf_usage = ECDF(vehicle_stats['avg_daily_mileage'].values)
        self.ecdf_maint = ECDF(vehicle_stats['maint_density'].values)
        self.fitted = True

        # 保存统计信息（用于调试）
        self.stats = {
            'n_vehicles': len(vehicle_stats),
            'avg_daily_mileage_mean': vehicle_stats['avg_daily_mileage'].mean(),
            'avg_daily_mileage_median': vehicle_stats['avg_daily_mileage'].median(),
            'maint_density_mean': vehicle_stats['maint_density'].mean(),
            'maint_density_median': vehicle_stats['maint_density'].median()
        }

        return self

    def predict_scores(self, mileage: float, days: int, maint_count: int) -> Tuple[float, float]:
        """
        预测行为得分

        Parameters:
        -----------
        mileage : float
            总里程
        days : int
            使用天数
        maint_count : int
            保养次数

        Returns:
        --------
        usage_score : float
            使用强度得分 (0-100)，越低表示使用越激烈
        maint_score : float
            保养规范度得分 (0-100)，越高表示保养越规范
        """
        if not self.fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        # 1. 计算指标
        avg_daily_mileage, maint_density = self._compute_metrics(mileage, days, maint_count)

        # 2. 使用 ECDF 计算得分
        # 使用强度得分: 100 * (1 - F(日里程)) - 跑得越凶，得分越低
        usage_percentile = self.ecdf_usage(avg_daily_mileage)
        usage_score = 100.0 * (1.0 - usage_percentile)

        # 保养规范度得分: 100 * F(保养密度) - 保养越勤，得分越高
        maint_percentile = self.ecdf_maint(maint_density)
        maint_score = 100.0 * maint_percentile

        return float(usage_score), float(maint_score)

    def _compute_metrics(
        self,
        mileage: float,
        days: int,
        maint_count: int
    ) -> Tuple[float, float]:
        """
        计算行为指标

        Parameters:
        -----------
        mileage : float
            总里程
        days : int
            使用天数
        maint_count : int
            保养次数

        Returns:
        --------
        avg_daily_mileage : float
            平均日里程
        maint_density : float
            保养密度 (每万公里保养次数)
        """
        # 处理异常值
        if days < 30:
            days = 30

        # avg_daily_mileage = 总里程 / 使用天数
        avg_daily_mileage = mileage / days

        # maint_density = 保养次数 / 总里程 * 10000 (每万公里保养次数)
        maint_density = maint_count / mileage * 10000 if mileage > 0 else 0.0

        return avg_daily_mileage, maint_density
