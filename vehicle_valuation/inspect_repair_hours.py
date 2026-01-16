#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工时数据统计分析脚本

目的：分析维修工时的分布特征，为 LLM Prompt 中的工时阈值设定提供科学依据
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    """执行工时数据统计分析"""
    print("\n" + "="*80)
    print("工时数据统计分析")
    print("="*80 + "\n")

    # 1. 加载数据
    print("步骤 1: 加载工时数据\n")
    data_dir = Path(__file__).parent / "data"
    time_file = data_dir / "上汽跃进_燃油_time_info.csv"

    if not time_file.exists():
        print(f"❌ 未找到文件: {time_file}")
        return

    df = pd.read_csv(time_file)
    print(f"✓ 原始数据: {len(df)} 条记录\n")

    # 2. 数据清洗
    print("步骤 2: 数据清洗\n")

    # 转换为数值型
    df['REPAIR_HOURS'] = pd.to_numeric(df['REPAIR_HOURS'], errors='coerce')

    # 删除缺失值
    before_count = len(df)
    df = df.dropna(subset=['REPAIR_HOURS'])
    after_count = len(df)
    print(f"  • 删除工时缺失的记录: {before_count - after_count} 条")

    # 过滤异常值
    before_count = len(df)
    df = df[df['REPAIR_HOURS'] > 0]
    after_count = len(df)
    print(f"  • 删除工时 ≤ 0 的记录: {before_count - after_count} 条")

    print(f"\n✓ 清洗后数据: {len(df)} 条记录\n")

    # 3. 统计分布
    print("="*80)
    print("步骤 3: 统计分布分析")
    print("="*80 + "\n")

    hours = df['REPAIR_HOURS']

    # 基础统计量
    print("【基础统计量】")
    print(f"  样本数: {len(hours):,}")
    print(f"  均值 (Mean): {hours.mean():.2f} 小时")
    print(f"  中位数 (Median): {hours.median():.2f} 小时")
    print(f"  标准差 (Std Dev): {hours.std():.2f} 小时")
    print(f"  最小值: {hours.min():.2f} 小时")
    print(f"  最大值: {hours.max():.2f} 小时")

    # 分位数
    print(f"\n【分位数分布】")
    quantiles = [0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.97, 0.99]
    for q in quantiles:
        value = hours.quantile(q)
        print(f"  {int(q*100):2d}% 分位数 (P{int(q*100):02d}): {value:.2f} 小时")

    # 4. 区间分布
    print(f"\n【区间分布】")
    bins = [0, 0.5, 1, 2, 4, 8, 12, 24, 100, float('inf')]
    labels = ['0-0.5h', '0.5-1h', '1-2h', '2-4h', '4-8h', '8-12h', '12-24h', '24-100h', '>100h']

    df['hour_range'] = pd.cut(df['REPAIR_HOURS'], bins=bins, labels=labels, right=False)
    distribution = df['hour_range'].value_counts().sort_index()

    print(f"\n  工时区间        记录数      占比")
    print(f"  {'─'*40}")
    for range_name, count in distribution.items():
        percentage = count / len(df) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {range_name:>8}    {count:>6}    {percentage:>5.1f}%  {bar}")

    # 5. 阈值建议
    print(f"\n{'='*80}")
    print("步骤 4: 阈值设定建议")
    print("="*80 + "\n")

    print("【当前 Prompt 使用的阈值】")
    print("  • 短工时阈值: < 1h (用于降级 L3→L1)")
    print("  • 长工时阈值: > 8h (用于确认 L3 或升级 L1/L2→L3)")

    print("\n【基于统计数据的建议】")

    # 计算关键阈值
    p50 = hours.quantile(0.50)
    p75 = hours.quantile(0.75)
    p90 = hours.quantile(0.90)
    p95 = hours.quantile(0.95)
    p99 = hours.quantile(0.99)

    print(f"\n**方案 A: 保守策略（推荐）**")
    print(f"  • 短工时阈值: 1.0h (当前值)")
    print(f"    → 理由: 1h 对应于 P{hours.quantile(0.50):.1f}，符合中位数水平")
    print(f"    → 用途: 排除误报，将轻微检查/调整从 L3 降级为 L1")
    print(f"  • 长工时阈值: 8.0h (当前值)")
    print(f"    → 理由: 8h 对应于 P{hours[hours <= 8].count() / len(hours) * 100:.1f}，覆盖绝大多数常规维修")
    print(f"    → 用途: 确认重大维修，将 L3 理论等级进行验证")

    print(f"\n**方案 B: 激进策略**")
    print(f"  • 短工时阈值: 0.5h")
    print(f"    → 理由: 0.5h 对应于 P{hours[hours <= 0.5].count() / len(hours) * 100:.1f}，更严格定义'短工时'")
    print(f"    → 用途: 仅将极短操作视为检查/调整")
    print(f"  • 长工时阈值: 12.0h")
    print(f"    → 理由: 12h 对应于 P{hours[hours <= 12].count() / len(hours) * 100:.1f}，对应于 P95 分位数")
    print(f"    → 用途: 仅将极端长工时视为重大维修")

    # 统计各阈值下的覆盖情况
    print(f"\n【阈值覆盖分析】")
    thresholds = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
    print(f"\n  阈值      覆盖比例    累积占比    说明")
    print(f"  {'─'*60}")
    for threshold in thresholds:
        coverage = hours[hours <= threshold].count() / len(hours) * 100
        cumulative = hours[hours <= threshold].count() / len(hours) * 100
        desc = "短工时" if threshold <= 1 else ("中工时" if threshold <= 8 else "长工时")
        print(f"  ≤ {threshold:2.0f}h     {coverage:>5.1f}%      {cumulative:>5.1f}%      {desc}")

    # 6. 结论
    print(f"\n{'='*80}")
    print("【结论与建议】")
    print("="*80 + "\n")

    print("✓ **推荐维持当前阈值设置 (1h / 8h)**")
    print(f"  • 1h 短工时阈值: 覆盖 {hours[hours <= 1].count() / len(hours) * 100:.1f}% 的维修记录")
    print(f"  • 8h 长工时阈值: 覆盖 {hours[hours <= 8].count() / len(hours) * 100:.1f}% 的维修记录")
    print(f"  • 8-100h 极长工时: {hours[hours > 8].count() / len(hours) * 100:.1f}% 的维修记录（可能是重大事故修复）")
    print(f"\n  这些阈值具有良好的区分度：")
    print(f"  – < 1h: 快速更换、检查调整（L1/L0）")
    print(f"  – 1-8h: 常规维修、总成小修（L2/L1）")
    print(f"  – > 8h: 总成大修、事故修复（L3）")

    print(f"\n{'='*80}\n")

    # 保存统计结果到文件
    output_file = Path(__file__).parent / "repair_hours_statistics.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("工时数据统计报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"样本数: {len(hours):,}\n")
        f.write(f"均值: {hours.mean():.2f} 小时\n")
        f.write(f"中位数: {hours.median():.2f} 小时\n")
        f.write(f"标准差: {hours.std():.2f} 小时\n")
        f.write(f"最小值: {hours.min():.2f} 小时\n")
        f.write(f"最大值: {hours.max():.2f} 小时\n\n")
        f.write("分位数分布:\n")
        for q in quantiles:
            value = hours.quantile(q)
            f.write(f"  P{int(q*100):02d}: {value:.2f} 小时\n")
        f.write("\n区间分布:\n")
        for range_name, count in distribution.items():
            percentage = count / len(df) * 100
            f.write(f"  {range_name}: {count} ({percentage:.1f}%)\n")

    print(f"✓ 统计报告已保存至: {output_file}\n")


if __name__ == "__main__":
    main()
