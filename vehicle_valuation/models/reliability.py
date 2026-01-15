#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆可靠性模型 - 故障率强度评估

功能:
- 基于 LLM 标注的维修记录严重程度评估车辆故障率强度
- 采用"先微观打标，再宏观聚合"的思路
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
from typing import Optional


class ReliabilityModel:
    """
    车辆可靠性评估模型

    基于维修记录的严重程度计算故障率强度，
    并通过与群体基准的对比进行评分。
    """

    # 权重标准常量
    WEIGHTS = {'L0': 0, 'L1': 1, 'L2': 5, 'L3': 20}

    def __init__(self):
        """初始化模型"""
        self.lambda_pop = None  # 群体基准故障率强度
        self.vehicle_profiles = None  # 车辆聚合数据 (DataFrame)
        self.fitted = False

    def fit(self, llm_df: pd.DataFrame, base_df: pd.DataFrame):
        """
        训练可靠性模型

        采用"先微观打标，再宏观聚合"的思路：
        1. 将 LLM 结果与基础信息通过 ID 关联
        2. 根据 Severity 映射权重
        3. 按 VIN 聚合计算每辆车的故障率强度
        4. 计算群体基准

        Parameters:
        -----------
        llm_df : pd.DataFrame
            LLM 分析结果，必须包含 ID, Severity 列
        base_df : pd.DataFrame
            基础信息表，必须包含 ID, VIN, REPAIR_MILEAGE 列
        """
        # 步骤 1: 关联 (Join) - 通过 ID 内连接
        print("  → 步骤 1: 关联 LLM 结果与基础信息...")
        merged_df = llm_df[['ID', 'Severity']].merge(
            base_df[['ID', 'VIN', 'REPAIR_MILEAGE']],
            on='ID',
            how='inner'  # 内连接，只保留两边都有的记录
        )

        if len(merged_df) == 0:
            raise ValueError("LLM 结果与基础信息没有匹配的记录，请检查 ID 列是否一致")

        print(f"     成功关联 {len(merged_df)} 条记录")

        # 步骤 2: 映射权重 (Map Weights)
        print("  → 步骤 2: 映射严重程度权重...")
        merged_df['weight'] = merged_df['Severity'].map(self.WEIGHTS)

        # 检查是否有未映射的 Severity
        unmapped = merged_df[merged_df['weight'].isna()]
        if len(unmapped) > 0:
            print(f"     ⚠ 警告: {len(unmapped)} 条记录的 Severity 值无法映射，将被忽略")
            print(f"     未映射的 Severity 值: {unmapped['Severity'].unique()}")
            merged_df = merged_df[merged_df['weight'].notna()]

        # 步骤 3: 车辆聚合 (Aggregation)
        print("  → 步骤 3: 按 VIN 聚合计算车辆指标...")
        vehicle_profiles = merged_df.groupby('VIN').agg({
            'weight': 'sum',              # 总故障分数
            'REPAIR_MILEAGE': 'max',     # 最大里程
            'ID': 'count'                # 记录数量
        }).reset_index()

        vehicle_profiles.rename(columns={
            'weight': 'total_fault_score',
            'REPAIR_MILEAGE': 'max_mileage',
            'ID': 'record_count'
        }, inplace=True)

        print(f"     聚合得到 {len(vehicle_profiles)} 辆车的数据")

        # 步骤 4: 计算指标
        print("  → 步骤 4: 计算故障率强度...")
        # 避免除零，设置最小里程为 1000 km
        vehicle_profiles['max_mileage'] = vehicle_profiles['max_mileage'].clip(lower=1000)

        # 计算故障率强度 Λ = total_fault_score / max_mileage
        vehicle_profiles['lambda'] = vehicle_profiles['total_fault_score'] / vehicle_profiles['max_mileage']

        # 计算群体平均强度 Λ_pop
        self.lambda_pop = vehicle_profiles['lambda'].mean()

        print(f"     群体基准 Λ_pop: {self.lambda_pop:.6f} /km")

        # 保存车辆聚合数据
        self.vehicle_profiles = vehicle_profiles

        # 保存统计信息
        self.stats = {
            'n_llm_records': len(merged_df),
            'n_vehicles': len(vehicle_profiles),
            'lambda_mean': vehicle_profiles['lambda'].mean(),
            'lambda_median': vehicle_profiles['lambda'].median(),
            'lambda_min': vehicle_profiles['lambda'].min(),
            'lambda_max': vehicle_profiles['lambda'].max(),
            'avg_records_per_vehicle': vehicle_profiles['record_count'].mean()
        }

        self.fitted = True

        return self

    def predict_score(self, vin: str) -> Optional[float]:
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

        Returns:
        --------
        score : float or None
            可靠性得分 (0-100)
            如果 VIN 不在训练集中，返回 None
        """
        if not self.fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        # 查找该 VIN 的车辆数据
        vehicle_row = self.vehicle_profiles[self.vehicle_profiles['VIN'] == vin]

        if vehicle_row.empty:
            # VIN 不在训练集中（这辆车没在 200 条样本里）
            return None

        # 获取故障率强度
        lambda_i = vehicle_row.iloc[0]['lambda']

        # 计算得分: 100 × exp(-0.693 × Λ_i / Λ_pop)
        score = 100.0 * np.exp(-0.693 * lambda_i / self.lambda_pop)

        return float(score)

    def get_vehicle_profile(self, vin: str) -> Optional[dict]:
        """
        获取车辆的详细数据

        Parameters:
        -----------
        vin : str
            车辆识别码

        Returns:
        --------
        profile : dict or None
            包含车辆详细信息的字典，如果 VIN 不存在则返回 None
        """
        if not self.fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        vehicle_row = self.vehicle_profiles[self.vehicle_profiles['VIN'] == vin]

        if vehicle_row.empty:
            return None

        row = vehicle_row.iloc[0]

        return {
            'vin': row['VIN'],
            'total_fault_score': row['total_fault_score'],
            'max_mileage': row['max_mileage'],
            'record_count': row['record_count'],
            'lambda': row['lambda']
        }
