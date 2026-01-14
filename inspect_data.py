#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查数据文件内容的脚本"""

import pandas as pd

# 定义文件路径
data_dir = "vehicle_valuation/data/"
files = {
    "baseinfo": "上汽跃进_燃油_baseinfo.csv",
    "parts_info": "上汽跃进_燃油_parts_info.csv",
    "time_info": "上汽跃进_燃油_time_info.csv"
}

# 尝试不同的编码方式
encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']

def read_csv_with_encoding(filepath):
    """尝试不同编码读取CSV"""
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, nrows=5)
            print(f"✓ 成功使用编码: {enc}")
            return df, enc
        except Exception as e:
            continue
    print(f"✗ 所有编码都失败")
    return None, None

# 检查每个文件
for name, filename in files.items():
    filepath = data_dir + filename
    print("\n" + "="*80)
    print(f"文件: {filename}")
    print("="*80)

    df, encoding = read_csv_with_encoding(filepath)

    if df is not None:
        print(f"\n使用编码: {encoding}")
        print(f"\n列名 ({len(df.columns)} 列):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")

        print(f"\n前 5 行数据:")
        print(df.to_string())

        # 检查特定列的数据格式
        print(f"\n数据类型:")
        print(df.dtypes)

        # 如果有日期或里程列，显示示例
        if '维修日期' in df.columns:
            print(f"\n'维修日期' 列示例值:")
            print(df['维修日期'].head())
        if '里程' in df.columns:
            print(f"\n'里程' 列示例值:")
            print(df['里程'].head())
    else:
        print("无法读取文件")

print("\n" + "="*80)
