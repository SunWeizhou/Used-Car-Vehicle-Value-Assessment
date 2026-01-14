#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块

功能:
- 加载 CSV 数据文件
- 清洗和转换数据
- 处理缺失值和异常值
"""

import pandas as pd
from pathlib import Path


def load_and_clean_data(data_dir="vehicle_valuation/data"):
    """
    加载并清洗二手车残值评估数据

    Parameters:
    -----------
    data_dir : str
        数据目录路径

    Returns:
    --------
    clean_base : pd.DataFrame
        基础信息表 (清洗后)
    clean_parts : pd.DataFrame
        配件信息表 (清洗后)
    clean_time : pd.DataFrame
        工时信息表 (清洗后)
    """
    data_path = Path(data_dir)

    # ============ 1. 读取基础信息表 (GBK编码) ============
    print("="*80)
    print("步骤 1: 读取数据")
    print("="*80)
    print("正在读取 baseinfo.csv...")
    df_base = pd.read_csv(
        data_path / "上汽跃进_燃油_baseinfo.csv",
        encoding='gbk'
    )
    print(f"  → 原始行数: {len(df_base):,}")

    # ============ 2. 读取配件信息表 (UTF-8编码) ============
    print("正在读取 parts_info.csv...")
    df_parts = pd.read_csv(
        data_path / "上汽跃进_燃油_parts_info.csv",
        encoding='utf-8'
    )
    print(f"  → 原始行数: {len(df_parts):,}")

    # ============ 3. 读取工时信息表 (UTF-8编码) ============
    print("正在读取 time_info.csv...")
    df_time = pd.read_csv(
        data_path / "上汽跃进_燃油_time_info.csv",
        encoding='utf-8'
    )
    print(f"  → 原始行数: {len(df_time):,}")

    # ============ 4. 清洗 baseinfo 表 ============
    print("\n" + "="*80)
    print("步骤 2: 清洗 baseinfo 表")
    print("="*80)

    original_count = len(df_base)

    # 4.1 删除重复的 ID.1 列 (如果存在)
    if 'ID.1' in df_base.columns:
        print("→ 删除重复列 'ID.1'")
        df_base = df_base.drop(columns=['ID.1'])

    # 4.2 去重
    before_dedup = len(df_base)
    df_base = df_base.drop_duplicates()
    after_dedup = len(df_base)
    if before_dedup > after_dedup:
        print(f"→ 去重: 删除 {before_dedup - after_dedup:,} 条重复记录")

    # 4.3 日期处理 - 使用 SETTLE_DATE 作为主要时间戳
    print("→ 日期处理: 转换 SETTLE_DATE 为 datetime")
    df_base['SETTLE_DATE'] = pd.to_datetime(
        df_base['SETTLE_DATE'],
        errors='coerce'  # 无法转换的设为 NaT
    )

    # 删除日期转换失败的行
    before_date = len(df_base)
    df_base = df_base.dropna(subset=['SETTLE_DATE'])
    after_date = len(df_base)
    if before_date > after_date:
        print(f"   删除 {before_date - after_date:,} 条日期无效的记录")

    # 4.4 里程处理
    print("→ 里程处理: 转换为数值型并过滤异常值")
    df_base['REPAIR_MILEAGE'] = pd.to_numeric(
        df_base['REPAIR_MILEAGE'],
        errors='coerce'
    )

    # 过滤逻辑: 只保留里程在 10 到 2,000,000 之间的记录
    before_mileage = len(df_base)
    df_base = df_base[
        (df_base['REPAIR_MILEAGE'] >= 10) &
        (df_base['REPAIR_MILEAGE'] <= 2000000) &
        (df_base['REPAIR_MILEAGE'].notna())
    ]
    after_mileage = len(df_base)
    if before_mileage > after_mileage:
        print(f"   删除 {before_mileage - after_mileage:,} 条里程异常的记录")
        print(f"   保留范围: 10 km - 2,000,000 km")

    # 4.5 ID 标准化 - 确保 ID 列是字符串格式
    print("→ ID 标准化: 转换 ID 列为字符串格式")
    df_base['ID'] = df_base['ID'].astype(str)

    # 输出清洗结果
    final_count = len(df_base)
    print(f"\n【Baseinfo 清洗结果】")
    print(f"  原始行数: {original_count:,}")
    print(f"  清洗后行数: {final_count:,}")
    print(f"  删除行数: {original_count - final_count:,} ({(original_count - final_count)/original_count*100:.2f}%)")

    # ============ 5. 清洗 parts_info 表 ============
    print("\n" + "="*80)
    print("步骤 3: 清洗 parts_info 表")
    print("="*80)

    original_parts = len(df_parts)

    # 5.1 ID 标准化 - 确保 RECORD_ID 是字符串格式
    print("→ ID 标准化: 转换 RECORD_ID 列为字符串格式")
    df_parts['RECORD_ID'] = df_parts['RECORD_ID'].astype(str)

    # 5.2 去重
    before_dedup_parts = len(df_parts)
    df_parts = df_parts.drop_duplicates()
    after_dedup_parts = len(df_parts)

    print(f"\n【Parts_info 清洗结果】")
    print(f"  原始行数: {original_parts:,}")
    print(f"  清洗后行数: {after_dedup_parts:,}")
    if before_dedup_parts > after_dedup_parts:
        print(f"  删除重复: {before_dedup_parts - after_dedup_parts:,} 条")

    # ============ 6. 清洗 time_info 表 ============
    print("\n" + "="*80)
    print("步骤 4: 清洗 time_info 表")
    print("="*80)

    original_time = len(df_time)

    # 6.1 ID 标准化 - 确保 RECORD_ID 是字符串格式
    print("→ ID 标准化: 转换 RECORD_ID 列为字符串格式")
    df_time['RECORD_ID'] = df_time['RECORD_ID'].astype(str)

    # 6.2 去重
    before_dedup_time = len(df_time)
    df_time = df_time.drop_duplicates()
    after_dedup_time = len(df_time)

    # 6.3 清洗工时数据 (删除负数)
    before_hours = len(df_time)
    df_time = df_time[df_time['REPAIR_HOURS'] >= 0]
    after_hours = len(df_time)

    print(f"\n【Time_info 清洗结果】")
    print(f"  原始行数: {original_time:,}")
    print(f"  清洗后行数: {after_hours:,}")
    if before_dedup_time > after_dedup_time:
        print(f"  删除重复: {before_dedup_time - after_dedup_time:,} 条")
    if before_hours > after_hours:
        print(f"  删除负工时: {before_hours - after_hours:,} 条")

    print("\n" + "="*80)
    print("✓ 数据清洗完成")
    print("="*80 + "\n")

    return df_base, df_parts, df_time


def print_data_summary(df_base, df_parts, df_time):
    """
    打印数据摘要信息

    Parameters:
    -----------
    df_base, df_parts, df_time : pd.DataFrame
        清洗后的数据表
    """
    print("="*80)
    print("数据摘要")
    print("="*80)

    print("\n【基础信息表 (baseinfo)】")
    print(f"记录数: {len(df_base):,}")
    print(f"内存占用: {df_base.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"时间范围: {df_base['SETTLE_DATE'].min()} 至 {df_base['SETTLE_DATE'].max()}")
    print(f"里程范围: {df_base['REPAIR_MILEAGE'].min():,.0f} km - {df_base['REPAIR_MILEAGE'].max():,.0f} km")

    print("\n【配件信息表 (parts_info)】")
    print(f"记录数: {len(df_parts):,}")
    print(f"内存占用: {df_parts.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"唯一配件数: {df_parts['PARTS_NAME'].nunique():,}")

    print("\n【工时信息表 (time_info)】")
    print(f"记录数: {len(df_time):,}")
    print(f"内存占用: {df_time.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"唯一维修项目数: {df_time['REPAIR_NAME'].nunique():,}")

    print("\n" + "="*80)


if __name__ == "__main__":
    # 测试代码
    df_base, df_parts, df_time = load_and_clean_data()
    print_data_summary(df_base, df_parts, df_time)
