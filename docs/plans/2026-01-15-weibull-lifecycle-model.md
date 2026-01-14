# Weibull Lifecycle Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Weibull distribution-based vehicle lifecycle scoring model to predict survival probability and calculate lifecycle scores based on mileage data.

**Architecture:** Statistical survival analysis model using Weibull distribution with maximum likelihood estimation (MLE). The model processes vehicle repair history to estimate shape (k) and scale (λ) parameters, then calculates survival probability S(t) = exp(-(t/λ)^k) for individual vehicles.

**Tech Stack:** pandas (data manipulation), numpy (numerical operations), scipy.optimize (MLE optimization)

---

## Task 1: Create lifecycle.py module structure

**Files:**
- Create: `vehicle_valuation/models/lifecycle.py`

**Step 1: Create module with docstring and imports**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆生命周期模型 - Weibull 分布

功能:
- 使用 Weibull 分布建模车辆寿命
- 极大似然估计 (MLE) 拟合参数
- 计算车辆生存概率和生命周期得分
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

    Parameters:
    -----------
    df_base : pd.DataFrame
        基础信息表 (清洗后)，需包含 VIN, REPAIR_MILEAGE, SETTLE_DATE 列

    Returns:
    --------
    result_df : pd.DataFrame
        包含 VIN, t (寿命里程), event (失效标记) 的 DataFrame
    """
    pass


class WeibullModel:
    """
    Weibull 分布寿命模型

    使用极大似然估计拟合 Weibull 分布参数，
    并预测车辆的生存概率。
    """

    def __init__(self):
        """初始化模型"""
        self.k = None  # 形状参数
        self.lambda_ = None  # 尺度参数

    def fit(self, t: np.ndarray, event: np.ndarray) -> 'WeibullModel':
        """
        使用 MLE 拟合 Weibull 参数

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
        pass

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
        pass

    def get_params(self) -> Dict[str, float]:
        """
        获取拟合参数

        Returns:
        --------
        params : dict
            包含 k 和 lambda_ 的字典
        """
        pass
```

**Step 2: Verify file created**

Run: `ls -la vehicle_valuation/models/lifecycle.py`
Expected: File exists

**Step 3: Commit**

```bash
git add vehicle_valuation/models/lifecycle.py
git commit -m "feat: create lifecycle module structure with WeibullModel class"
```

---

## Task 2: Implement prepare_weibull_data function

**Files:**
- Modify: `vehicle_valuation/models/lifecycle.py:24-60`

**Step 1: Write failing test for prepare_weibull_data**

Create test file: `tests/test_lifecycle.py`

```python
import pytest
import pandas as pd
from datetime import datetime, timedelta
from vehicle_valuation.models.lifecycle import prepare_weibull_data


def test_prepare_weibull_data_basic():
    """Test basic data preparation functionality"""
    # Create sample data
    data = {
        'VIN': ['VIN1', 'VIN1', 'VIN2', 'VIN3'],
        'REPAIR_MILEAGE': [10000, 20000, 50000, 30000],
        'SETTLE_DATE': pd.to_datetime([
            '2020-01-01', '2022-01-01', '2021-01-01', '2023-01-01'
        ])
    }
    df = pd.DataFrame(data)

    # Mock current date as 2025-01-01
    # Note: This test will fail because function is not implemented
    result = prepare_weibull_data(df)

    # Verify structure
    assert 'VIN' in result.columns
    assert 't' in result.columns
    assert 'event' in result.columns
    assert len(result) == 3  # 3 unique VINs

    # Verify VIN1 has max mileage 20000
    vin1_row = result[result['VIN'] == 'VIN1'].iloc[0]
    assert vin1_row['t'] == 20000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_lifecycle.py::test_prepare_weibull_data_basic -v`
Expected: FAIL or error (function not implemented)

**Step 3: Implement prepare_weibull_data**

Replace the `pass` in `prepare_weibull_data` with:

```python
def prepare_weibull_data(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    准备 Weibull 模型数据

    按 VIN 分组，计算每辆车的最大里程和最后出现日期，
    并根据时间间隔标记是否失效。

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

    # 3. 定义失效事件
    # 如果最后出现日期距离 current_date 超过 730 天 (2年)，则认为已报废
    vehicle_stats['days_since_last_seen'] = (current_date - vehicle_stats['last_seen_date']).dt.days
    vehicle_stats['event'] = (vehicle_stats['days_since_last_seen'] > 730).astype(int)

    # 4. 返回结果
    result_df = vehicle_stats[['VIN', 't', 'event']].copy()

    return result_df
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_lifecycle.py::test_prepare_weibull_data_basic -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_lifecycle.py vehicle_valuation/models/lifecycle.py
git commit -m "feat: implement prepare_weibull_data function with grouping and event labeling"
```

---

## Task 3: Implement Weibull negative log-likelihood

**Files:**
- Modify: `vehicle_valuation/models/lifecycle.py:76-100`

**Step 1: Write test for MLE fitting**

```python
def test_weibull_fit():
    """Test Weibull model fitting"""
    from vehicle_valuation.models.lifecycle import WeibullModel
    import numpy as np

    # Create synthetic data with known parameters
    np.random.seed(42)
    k_true, lambda_true = 2.0, 100000.0

    # Generate samples
    n = 100
    t = np.random.weibull(k_true, n) * lambda_true

    # Random censoring (30% censored)
    event = np.random.binomial(1, 0.7, n)

    # Fit model
    model = WeibullModel()
    model.fit(t, event)

    # Check parameters are reasonable
    params = model.get_params()
    assert params['k'] is not None
    assert params['lambda_'] is not None
    assert params['k'] > 0
    assert params['lambda_'] > 0

    print(f"Fitted k: {params['k']:.2f} (true: {k_true})")
    print(f"Fitted λ: {params['lambda_']:.0f} (true: {lambda_true})")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_lifecycle.py::test_weibull_fit -v`
Expected: FAIL (fit method not implemented)

**Step 3: Implement negative log-likelihood function**

Add this helper function before the `WeibullModel` class:

```python
def _weibull_neg_log_likelihood(params: np.ndarray, t: np.ndarray, event: np.ndarray) -> float:
    """
    Weibull 分布的负对数似然函数

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

    # 对数似然
    # L = Σ[event[i] * log(f(t[i])) + (1-event[i]) * log(S(t[i]))]

    # 计算 (t/λ)^k
    z = (t / lambda_) ** k

    # log(f(t)) = log(k/λ) + (k-1)*log(t/λ) - z
    log_f = np.log(k / lambda_) + (k - 1) * np.log(t / lambda_ + 1e-10) - z

    # log(S(t)) = -z
    log_s = -z

    # 对数似然
    log_likelihood = np.sum(event * log_f + (1 - event) * log_s)

    return -log_likelihood
```

**Step 4: Run test**

Run: `pytest tests/test_lifecycle.py::test_weibull_fit -v`
Expected: FAIL (fit method still not implemented, but neg_log_likelihood exists)

**Step 5: Commit**

```bash
git add vehicle_valuation/models/lifecycle.py
git commit -m "feat: implement Weibull negative log-likelihood function"
```

---

## Task 4: Implement WeibullModel.fit method

**Files:**
- Modify: `vehicle_valuation/models/lifecycle.py:76-110`

**Step 1: Implement fit method**

Replace the `pass` in `fit` method with:

```python
def fit(self, t: np.ndarray, event: np.ndarray) -> 'WeibullModel':
    """
    使用 MLE 拟合 Weibull 参数

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

    # 优化
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

    # 保存参数
    self.k = result.x[0]
    self.lambda_ = result.x[1]

    return self
```

**Step 2: Run test to verify fit works**

Run: `pytest tests/test_lifecycle.py::test_weibull_fit -v`
Expected: PASS (parameters should be close to true values)

**Step 3: Commit**

```bash
git add vehicle_valuation/models/lifecycle.py tests/test_lifecycle.py
git commit -m "feat: implement WeibullModel.fit with MLE optimization"
```

---

## Task 5: Implement WeibullModel.predict_score method

**Files:**
- Modify: `vehicle_valuation/models/lifecycle.py:112-135`

**Step 1: Write test for predict_score**

```python
def test_predict_score():
    """Test lifecycle score prediction"""
    from vehicle_valuation.models.lifecycle import WeibullModel

    # Create and fit model
    model = WeibullModel()
    model.k = 2.0
    model.lambda_ = 100000.0

    # Test prediction
    # At t = 0, score should be 100
    assert abs(model.predict_score(0) - 100.0) < 0.01

    # At t = lambda, score = 100 * exp(-1) ≈ 36.79
    score_at_lambda = model.predict_score(100000.0)
    assert abs(score_at_lambda - 36.79) < 0.1

    # At t = 2*lambda, score = 100 * exp(-4) ≈ 1.83
    score_at_2lambda = model.predict_score(200000.0)
    assert abs(score_at_2lambda - 1.83) < 0.1

    print(f"Score at t=0: {model.predict_score(0):.2f}")
    print(f"Score at t=λ: {score_at_lambda:.2f}")
    print(f"Score at t=2λ: {score_at_2lambda:.2f}")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_lifecycle.py::test_predict_score -v`
Expected: FAIL (predict_score not implemented)

**Step 3: Implement predict_score method**

Replace the `pass` in `predict_score` with:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_lifecycle.py::test_predict_score -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vehicle_valuation/models/lifecycle.py tests/test_lifecycle.py
git commit -m "feat: implement WeibullModel.predict_score for lifecycle scoring"
```

---

## Task 6: Implement WeibullModel.get_params method

**Files:**
- Modify: `vehicle_valuation/models/lifecycle.py:137-155`

**Step 1: Write test for get_params**

```python
def test_get_params():
    """Test parameter retrieval"""
    from vehicle_valuation.models.lifecycle import WeibullModel

    model = WeibullModel()
    model.k = 2.5
    model.lambda_ = 150000.0

    params = model.get_params()

    assert params['k'] == 2.5
    assert params['lambda_'] == 150000.0
    assert isinstance(params, dict)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_lifecycle.py::test_get_params -v`
Expected: FAIL (get_params not implemented)

**Step 3: Implement get_params method**

Replace the `pass` in `get_params` with:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_lifecycle.py::test_get_params -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vehicle_valuation/models/lifecycle.py tests/test_lifecycle.py
git commit -m "feat: implement WeibullModel.get_params for parameter retrieval"
```

---

## Task 7: Update main.py to integrate Weibull model

**Files:**
- Modify: `vehicle_valuation/main.py:14-86`

**Step 1: Read current main.py structure**

Run: `head -n 30 vehicle_valuation/main.py`

**Step 2: Add import and Weibull modeling code**

After the data validation section (before line 86), add:

```python
# 3. Weibull 生命周期建模
print("\n" + "="*80)
print("步骤 3: Weibull 生命周期建模")
print("="*80)

from models.lifecycle import prepare_weibull_data, WeibullModel
import numpy as np

# 3.1 准备 Weibull 数据
print("\n【数据准备】")
weibull_df = prepare_weibull_data(df_base)
print(f"  车辆数量: {len(weibull_df):,}")
print(f"  已失效车辆: {weibull_df['event'].sum():,}")
print(f"  存活车辆: {(weibull_df['event'] == 0).sum():,}")

# 3.2 拟合 Weibull 模型
print("\n【模型拟合】")
model = WeibullModel()
model.fit(
    t=weibull_df['t'].values,
    event=weibull_df['event'].values
)

# 3.3 输出参数
params = model.get_params()
print(f"\n【拟合参数】")
print(f"  形状参数 k:  {params['k']:.4f}")
print(f"  尺度参数 λ:  {params['lambda_']:2f} km")
print(f"\n参数解释:")
print(f"  - k < 1: 故障率随时间下降 (早期失效)")
print(f"  - k = 1: 故障率恒定 (随机失效, 指数分布)")
print(f"  - k > 1: 故障率随时间上升 (磨损失效)")

# 3.4 案例展示
print("\n【案例展示】")
np.random.seed(42)
sample_vins = weibull_df.sample(5)

for idx, row in sample_vins.iterrows():
    vin = row['VIN']
    t_current = row['t']
    event = int(row['event'])
    event_label = "已报废" if event == 1 else "存活"

    score = model.predict_score(t_current)

    print(f"\n车辆 {vin[:8]}...")
    print(f"  当前里程: {t_current:,.0f} km")
    print(f"  状态: {event_label}")
    print(f"  生命周期得分: {score:.2f} / 100")

print("\n" + "="*80)
print("✓ Weibull 生命周期建模完成！")
print("="*80 + "\n")
```

**Step 3: Test main.py execution**

Run: `cd vehicle_valuation && python main.py`

Expected output should include:
- Data preparation statistics (vehicle count, failed/surviving)
- Fitted parameters (k and lambda)
- 5 sample vehicles with their lifecycle scores

**Step 4: Commit**

```bash
git add vehicle_valuation/main.py
git commit -m "feat: integrate Weibull lifecycle modeling into main pipeline"
```

---

## Task 8: End-to-end integration test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vehicle_valuation.utils.preprocessing import load_and_clean_data
from vehicle_valuation.models.lifecycle import prepare_weibull_data, WeibullModel


def test_full_pipeline():
    """Test full pipeline from data loading to scoring"""
    # 1. Load data
    df_base, df_parts, df_time = load_and_clean_data("vehicle_valuation/data")

    # 2. Prepare Weibull data
    weibull_df = prepare_weibull_data(df_base)

    # 3. Fit model
    model = WeibullModel()
    model.fit(
        t=weibull_df['t'].values,
        event=weibull_df['event'].values
    )

    # 4. Verify parameters
    params = model.get_params()
    assert params['k'] > 0
    assert params['lambda_'] > 0

    # 5. Test predictions on sample
    sample = weibull_df.sample(10)
    for _, row in sample.iterrows():
        score = model.predict_score(row['t'])
        assert 0 <= score <= 100

    print(f"\n✓ Integration test passed!")
    print(f"  Fitted k: {params['k']:.4f}")
    print(f"  Fitted λ: {params['lambda_']:.2f} km")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
```

**Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v -s`

Expected: PASS with output showing fitted parameters

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test for Weibull pipeline"
```

---

## Task 9: Add documentation and usage examples

**Files:**
- Create: `vehicle_valuation/models/LIFECYCLE_GUIDE.md`

**Step 1: Create documentation**

```markdown
# 车辆生命周期模型使用指南

## 概述

`lifecycle.py` 实现了基于 Weibull 分布的车辆生命周期建模，用于评估车辆的生存概率和生命周期得分。

## 理论基础

### Weibull 分布

Weibull 分布是可靠性工程中常用的寿命分布模型：

- **概率密度函数 (PDF)**: f(t) = (k/λ) × (t/λ)^(k-1) × exp(-(t/λ)^k)
- **生存函数**: S(t) = exp(-(t/λ)^k)
- **累积分布函数**: F(t) = 1 - exp(-(t/λ)^k)

其中：
- **k (形状参数)**: 决定分布形状
  - k < 1: 早期失效期 (故障率下降)
  - k = 1: 随机失效期 (故障率恒定，等价于指数分布)
  - k > 1: 磨损失效期 (故障率上升)

- **λ (尺度参数)**: 特征寿命，当 t = λ 时，约 63.2% 的车辆已失效

## 使用方法

### 基本流程

```python
from vehicle_valuation.utils.preprocessing import load_and_clean_data
from vehicle_valuation.models.lifecycle import prepare_weibull_data, WeibullModel

# 1. 加载数据
df_base, df_parts, df_time = load_and_clean_data("vehicle_valuation/data")

# 2. 准备 Weibull 数据
weibull_df = prepare_weibull_data(df_base)

# 3. 拟合模型
model = WeibullModel()
model.fit(
    t=weibull_df['t'].values,
    event=weibull_df['event'].values
)

# 4. 查看参数
params = model.get_params()
print(f"k = {params['k']:.4f}")
print(f"λ = {params['lambda_']:.2f} km")

# 5. 预测得分
score = model.predict_score(t_current=50000)
print(f"生命周期得分: {score:.2f}")
```

### 单辆车评分

```python
# 对单辆车进行评分
vin = "your_vin_here"
vehicle_data = weibull_df[weibull_df['VIN'] == vin]

if not vehicle_data.empty:
    t_current = vehicle_data.iloc[0]['t']
    score = model.predict_score(t_current)
    print(f"车辆 {vin} 的生命周期得分: {score:.2f}")
```

### 批量评分

```python
# 对所有车辆批量评分
weibull_df['lifecycle_score'] = weibull_df['t'].apply(
    lambda t: model.predict_score(t)
)

# 查看得分分布
print(weibull_df['lifecycle_score'].describe())

# 筛选高价值车辆 (得分 > 80)
high_value = weibull_df[weibull_df['lifecycle_score'] > 80]
print(f"高价值车辆数量: {len(high_value)}")
```

## 数据准备说明

### prepare_weibull_data 函数

该函数执行以下操作：

1. **按 VIN 分组**: 将同一辆车的多条维修记录合并
2. **计算寿命 t**: 取最大里程作为车辆寿命观测值
3. **标记失效事件**:
   - 如果车辆最后出现日期距离数据集最大日期超过 730 天 (2年)，标记为 `event=1` (已报废)
   - 否则标记为 `event=0` (存活/右截断)

### 失效定义说明

730 天 (2年) 的阈值可根据实际业务场景调整：
- **更严格的阈值** (如 365 天): 更多车辆被标记为失效
- **更宽松的阈值** (如 1095 天): 更多车辆被标记为存活

修改方法:

```python
# 在 prepare_weibull_data 函数中
threshold_days = 365  # 改为你想要的阈值
vehicle_stats['event'] = (vehicle_stats['days_since_last_seen'] > threshold_days).astype(int)
```

## 模型解释

### 生命周期得分

- **得分范围**: 0-100
- **得分 = 100 × S(t)**: 生存概率的百分比表示
- **含义**:
  - 90-100 分: 非常健康，处于生命周期的早期
  - 70-90 分: 健康，处于生命周期的中期
  - 50-70 分: 一般，进入生命周期的后期
  - 30-50 分: 较差，接近特征寿命
  - 0-30 分: 很差，已超过特征寿命

### 示例解读

假设拟合参数: k = 2.0, λ = 150,000 km

- t = 50,000 km → score ≈ 89 分: 健康
- t = 100,000 km → score ≈ 51 分: 一般
- t = 150,000 km → score ≈ 37 分: 接近特征寿命
- t = 200,000 km → score ≈ 19 分: 已超过特征寿命

## 注意事项

1. **数据质量**: 确保 REPAIR_MILEAGE 和 SETTLE_DATE 列无缺失值
2. **VIN 唯一性**: 同一 VIN 的多条记录会被合并
3. **右截断**: 存活车辆的数据是右截断的，模型已通过似然函数处理
4. **里程单位**: 确保里程单位一致 (本系统使用 km)

## 参考文献

- Weibull 分布: https://en.wikipedia.org/wiki/Weibull_distribution
- 生存分析: https://en.wikipedia.org/wiki/Survival_analysis
- 极大似然估计: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
```

**Step 2: Commit**

```bash
git add vehicle_valuation/models/LIFECYCLE_GUIDE.md
git commit -m "docs: add comprehensive lifecycle model usage guide"
```

---

## Task 10: Final verification and cleanup

**Files:**
- Modify: `vehicle_valuation/models/__init__.py`

**Step 1: Update models/__init__.py to expose lifecycle module**

```python
"""
Models package for vehicle valuation system
"""

from .lifecycle import prepare_weibull_data, WeibullModel

__all__ = ['prepare_weibull_data', 'WeibullModel']
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`

Expected: All tests PASS

**Step 3: Run main.py to verify end-to-end**

Run: `cd vehicle_valuation && python main.py`

Expected: Complete output with data validation and Weibull modeling results

**Step 4: Final commit**

```bash
git add vehicle_valuation/models/__init__.py
git commit -m "feat: expose lifecycle module in models package"
```

---

## Summary

This plan implements a complete Weibull distribution-based vehicle lifecycle model with:

1. **Data preparation**: Groups repair records by VIN and labels failure events
2. **Model fitting**: Maximum likelihood estimation for shape (k) and scale (λ) parameters
3. **Scoring**: Calculates lifecycle scores (0-100) based on survival probability
4. **Testing**: Unit tests, integration tests, and end-to-end validation
5. **Documentation**: Comprehensive usage guide with examples

The model provides statistical rigor for vehicle valuation by quantifying the remaining useful life of commercial vehicles based on their mileage history.
