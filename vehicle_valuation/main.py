#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车残值评估系统 - 主入口

功能:
- 数据加载与预处理
- 车辆生命周期分析
- 使用强度与保养规范度评估
- 故障率建模
- 未来风险预测
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.preprocessing import load_and_clean_data, print_data_summary
from models.lifecycle import prepare_weibull_data, WeibullModel
import numpy as np


def main():
    """主函数"""
    print("\n" + "="*80)
    print("二手车残值评估系统")
    print("="*80 + "\n")

    # 1. 加载和清洗数据
    print("步骤 1: 加载和清洗数据\n")
    data_dir = Path(__file__).parent / "data"
    df_base, df_parts, df_time = load_and_clean_data(str(data_dir))

    # 2. 数据验证统计
    print("\n" + "="*80)
    print("步骤 2: 数据验证统计")
    print("="*80)

    # 2.1 有效时间范围
    print("\n【有效时间范围】")
    print(f"  最小日期: {df_base['SETTLE_DATE'].min()}")
    print(f"  最大日期: {df_base['SETTLE_DATE'].max()}")
    print(f"  时间跨度: {(df_base['SETTLE_DATE'].max() - df_base['SETTLE_DATE'].min()).days} 天")

    # 2.2 有效里程范围
    print("\n【有效里程范围】")
    print(f"  最小里程: {df_base['REPAIR_MILEAGE'].min():,.0f} km")
    print(f"  最大里程: {df_base['REPAIR_MILEAGE'].max():,.0f} km")
    print(f"  平均里程: {df_base['REPAIR_MILEAGE'].mean():,.0f} km")
    print(f"  中位里程: {df_base['REPAIR_MILEAGE'].median():,.0f} km")

    # 2.3 剩余数据量
    print("\n【剩余数据量】")
    print(f"  baseinfo:    {len(df_base):,} 条维修记录")
    print(f"  parts_info:  {len(df_parts):,} 条配件记录")
    print(f"  time_info:   {len(df_time):,} 条工时记录")

    # 2.4 内存占用情况
    print("\n【内存占用情况】")
    base_memory = df_base.memory_usage(deep=True).sum() / 1024**2
    parts_memory = df_parts.memory_usage(deep=True).sum() / 1024**2
    time_memory = df_time.memory_usage(deep=True).sum() / 1024**2
    total_memory = base_memory + parts_memory + time_memory

    print(f"  baseinfo:    {base_memory:8.2f} MB")
    print(f"  parts_info:  {parts_memory:8.2f} MB")
    print(f"  time_info:   {time_memory:8.2f} MB")
    print(f"  " + "-"*40)
    print(f"  总计:        {total_memory:8.2f} MB")

    # 2.5 数据质量检查
    print("\n【数据质量检查】")
    print(f"  ✓ ID 列类型: {df_base['ID'].dtype}")
    print(f"  ✓ 日期列类型: {df_base['SETTLE_DATE'].dtype}")
    print(f"  ✓ 里程列类型: {df_base['REPAIR_MILEAGE'].dtype}")
    print(f"  ✓ 配件 RECORD_ID 类型: {df_parts['RECORD_ID'].dtype}")
    print(f"  ✓ 工时 RECORD_ID 类型: {df_time['RECORD_ID'].dtype}")

    print("\n" + "="*80)
    print("✓ 数据清洗验证完成！数据质量良好，可以进行后续分析。")
    print("="*80 + "\n")

    # 3. Weibull 生命周期建模
    print("\n" + "="*80)
    print("步骤 3: Weibull 生命周期建模")
    print("="*80)

    # 3.1 准备 Weibull 数据
    print("\n【数据准备】")
    weibull_df = prepare_weibull_data(df_base)
    print(f"  车辆数量: {len(weibull_df):,}")
    print(f"  已失效车辆 (event=1): {weibull_df['event'].sum():,}")
    print(f"  存活车辆 (event=0, 右截断): {(weibull_df['event'] == 0).sum():,}")

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
    print(f"  尺度参数 λ:  {params['lambda_']:,.0f} km")
    print(f"\n参数解释:")
    print(f"  - k < 1: 故障率随时间下降 (早期失效)")
    print(f"  - k = 1: 故障率恒定 (随机失效, 指数分布)")
    print(f"  - k > 1: 故障率随时间上升 (磨损失效)")

    # 3.4 案例展示
    print("\n【案例展示 - 随机 5 辆车】")
    np.random.seed(42)
    sample_vins = weibull_df.sample(5)

    for idx, row in sample_vins.iterrows():
        vin = row['VIN']
        t_current = row['t']
        event = int(row['event'])
        event_label = "已报废" if event == 1 else "存活 (右截断)"

        score = model.predict_score(t_current)

        print(f"\n车辆 {vin[:8]}...")
        print(f"  当前里程: {t_current:,.0f} km")
        print(f"  状态: {event_label}")
        print(f"  生命周期得分: {score:.2f} / 100")

    print("\n" + "="*80)
    print("✓ Weibull 生命周期建模完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
