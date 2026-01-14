#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 结构化处理快速测试
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent / "vehicle_valuation"
sys.path.insert(0, str(project_root))

from utils.preprocessing import load_and_clean_data
from utils.llm_structuring import process_sample_batch


def main():
    """主函数"""
    print("\n" + "="*80)
    print("LLM 结构化处理测试")
    print("="*80 + "\n")

    # 1. 加载数据
    print("步骤 1: 加载数据\n")
    data_dir = project_root / "data"
    df_base, df_parts, df_time = load_and_clean_data(str(data_dir))

    # 2. 设置 API Key
    api_key = "sk-b64c19326e4c403089c1fb90dce96aca"

    # 3. 处理样本数据（200条）
    print("\n步骤 2: 处理样本数据（200条记录）")
    print("-"*80)
    print("基于中国国家标准 GB/T 进行分级")
    print("  - GB/T 30323-2013 (二手车鉴定评估技术规范)")
    print("  - GB/T 5624-2005 (汽车维修术语)")
    print("  - GB/T 18344-2016 (汽车维护、检测、诊断技术规范)")
    print("-"*80 + "\n")

    results_df = process_sample_batch(
        base_df=df_base,
        parts_df=df_parts,
        time_df=df_time,
        api_key=api_key,
        sample_size=200  # 处理 200 条
    )

    # 4. 显示详细结果（前10条）
    print("\n步骤 3: 详细结果（前10条）")
    print("="*80)
    for idx, row in results_df.head(10).iterrows():
        print(f"\n记录 {idx + 1}:")
        print(f"  ID: {row['ID']}")
        print(f"  事件类型: {row['Event_Type']}")
        print(f"  受损系统: {row['System']}")
        print(f"  严重程度: {row['Severity']}")
        print(f"  理由: {row['Reasoning']}")

    print("\n" + "="*80)
    print("✓ 测试完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
