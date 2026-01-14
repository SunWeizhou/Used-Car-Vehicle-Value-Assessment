#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆生命周期模型 - Weibull 分布

功能:
- 使用 Weibull 分布建模车辆寿命
- 极大似然估计 (MLE) 拟合参数
- 计算车辆生存概率和生命周期得分

理论基础:
- Weibull 分布生存函数: S(t) = exp(-(t/λ)^k)
- 概率密度函数: f(t) = (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)
- 对数似然: L = Σ[event[i] * log(f(t[i])) + (1-event[i]) * log(S(t[i]))]
  其中 event=1 表示已失效（使用 f(t)），event=0 表示右截断（使用 S(t)）
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict


def prepare_weibull_data(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    准备 Weibull 模型数据

    按 VIN 分组，计算每辆车的最大里程和最后出现日期，
    并根据时间间隔标记是否失效。

    关键：正确处理右截断 (Right Censoring)
    - 只有最后出现日期距离当前超过 730 天的车辆标记为 event=1 (已失效)
    - 其他车辆标记为 event=0 (右截断/存活)，这些样本对似然函数仍有贡献

    Parameters:
    -----------
    df_base : pd.DataFrame
        基础信息表 (清洗后)，需包含 VIN, REPAIR_MILEAGE, SETTLE_DATE 列

    Returns:
    --------
    result_df : pd.DataFrame
        包含 VIN, t (寿命里程), event (失效标记) 的 DataFrame
    """
    # 确保 SETTLE_DATE 是 datetime 类型
    df_base = df_base.copy()
    df_base['SETTLE_DATE'] = pd.to_datetime(df_base['SETTLE_DATE'])

    # 1. 获取数据集全局最大日期作为当前日期
    current_date = df_base['SETTLE_DATE'].max()

    # 2. 按 VIN 分组聚合
    vehicle_stats = df_base.groupby('VIN').agg({
        'REPAIR_MILEAGE': 'max',  # 最大里程作为寿命 t
        'SETTLE_DATE': 'max'      # 最后出现日期
    }).reset_index()

    vehicle_stats.rename(columns={
        'REPAIR_MILEAGE': 't',
        'SETTLE_DATE': 'last_seen_date'
    }, inplace=True)

    # 3. 定义失效事件 - 关键：正确处理右截断
    # 如果最后出现日期距离 current_date 超过 730 天 (2年)，则认为已报废
    # 其他车辆标记为 event=0 (右截断/存活)，这些样本在 MLE 中使用 S(t) 而非 f(t)
    vehicle_stats['days_since_last_seen'] = (current_date - vehicle_stats['last_seen_date']).dt.days
    vehicle_stats['event'] = (vehicle_stats['days_since_last_seen'] > 730).astype(int)

    # 4. 返回结果
    result_df = vehicle_stats[['VIN', 't', 'event']].copy()

    return result_df


class WeibullModel:
    """
    Weibull 分布寿命模型

    使用极大似然估计拟合 Weibull 分布参数，
    并预测车辆的生存概率。
    """

    def __init__(self):
        """初始化模型"""
        self.k = None  # 形状参数 (shape parameter)
        self.lambda_ = None  # 尺度参数 (scale parameter)

    def fit(self, t: np.ndarray, event: np.ndarray) -> 'WeibullModel':
        """
        使用 MLE 拟合 Weibull 参数

        极大似然估计：
        最小化负对数似然: -ln(L) = -Σ[event[i] * ln(f(t[i])) + (1-event[i]) * ln(S(t[i]))]

        其中:
        - event=1 (失效): 使用概率密度 f(t)
        - event=0 (右截断): 使用生存函数 S(t)

        Parameters:
        -----------
        t : np.ndarray
            寿命观测值 (里程)
        event : np.ndarray
            失效标记 (1=失效, 0=右截断)

        Returns:
        --------
        self : WeibullModel
        """
        # 初始参数猜测
        # k: 形状参数，通常在 1-5 之间
        # lambda: 尺度参数，使用数据的平均值作为初始值
        k_init = 2.0
        lambda_init = np.mean(t)

        # 使用 scipy.optimize.minimize 进行 MLE 优化
        result = minimize(
            _weibull_neg_log_likelihood,
            x0=[k_init, lambda_init],
            args=(t, event),
            method='L-BFGS-B',
            bounds=[(0.1, 10.0), (1000.0, None)],  # k: [0.1, 10], lambda: [1000, inf]
            options={'maxiter': 1000}
        )

        if not result.success:
            raise RuntimeError(f"MLE 优化失败: {result.message}")

        # 保存拟合参数
        self.k = result.x[0]
        self.lambda_ = result.x[1]

        return self

    def predict_score(self, t_current: float) -> float:
        """
        计算生命周期得分

        得分 = 100 * S(t) = 100 * exp(-(t/λ)^k)

        Parameters:
        -----------
        t_current : float
            当前里程

        Returns:
        --------
        score : float
            生命周期得分 (0-100)
        """
        if self.k is None or self.lambda_ is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        # 计算生存概率 S(t) = exp(-(t/λ)^k)
        survival_prob = np.exp(-((t_current / self.lambda_) ** self.k))

        # 转换为 0-100 分数
        score = 100.0 * survival_prob

        return float(score)

    def get_params(self) -> Dict[str, float]:
        """
        获取拟合参数

        Returns:
        --------
        params : dict
            包含 k 和 lambda_ 的字典
        """
        if self.k is None or self.lambda_ is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")

        return {
            'k': self.k,
            'lambda_': self.lambda_
        }


def _weibull_neg_log_likelihood(params: np.ndarray, t: np.ndarray, event: np.ndarray) -> float:
    """
    Weibull 分布的负对数似然函数

    正确处理右截断数据：
    - 失效样本 (event=1): 贡献 f(t) = (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)
    - 截断样本 (event=0): 贡献 S(t) = exp(-(t/λ)^k)

    对数似然:
    ln(L) = Σ[event[i] * ln(f(t[i])) + (1-event[i]) * ln(S(t[i]))]

    这正是生存分析中处理右截断的标准方法！

    Parameters:
    -----------
    params : np.ndarray
        [k, lambda] 参数数组
    t : np.ndarray
        寿命观测值
    event : np.ndarray
        失效标记 (1=失效, 0=右截断)

    Returns:
    --------
    nll : float
        负对数似然值
    """
    k, lambda_ = params

    # 避免除零和负值
    if k <= 0 or lambda_ <= 0:
        return np.inf

    # Weibull 分布函数
    # PDF: f(t) = (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)
    # CDF: F(t) = 1 - exp(-(t/λ)^k)
    # Survival: S(t) = exp(-(t/λ)^k)

    # 计算 (t/λ)^k，添加小常数避免 log(0)
    z = (t / lambda_) ** k

    # log(f(t)) = log(k/λ) + (k-1)*log(t/λ) - z
    log_f = np.log(k / lambda_) + (k - 1) * np.log(t / lambda_ + 1e-10) - z

    # log(S(t)) = -z
    log_s = -z

    # 对数似然：event=1 用 log(f(t)), event=0 用 log(S(t))
    # 这就是处理右截断的标准公式！
    log_likelihood = np.sum(event * log_f + (1 - event) * log_s)

    return -log_likelihood
