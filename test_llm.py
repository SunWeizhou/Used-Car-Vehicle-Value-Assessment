#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 结构化处理测试脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent / "vehicle_valuation"
sys.path.insert(0, str(project_root))

from utils.preprocessing import load_and_clean_data
from utils.llm_structuring import process_sample_batch, test_llm_connection


def main():
    """主函数"""
    print("\n" + "="*80)
    print("LLM 结构化处理测试")
    print("="*80 + "\n")

    # 1. 加载数据
    print("步骤 1: 加载数据\n")
    data_dir = project_root / "data"
    df_base, df_parts, df_time = load_and_clean_data(str(data_dir))

    # 2. 测试 API 连接
    print("\n步骤 2: 测试 API 连接")
    print("-"*80)

    # 注意：这里需要替换为你的真实 API Key
    # 方式 1: 直接在代码中设置（不推荐，仅用于测试）
    # api_key = "sk-your-deepseek-api-key-here"

    # 方式 2: 从环境变量读取（推荐）
    import os
    api_key = os.getenv('DEEPSEEK_API_KEY')

    if not api_key:
        print("\n⚠ 请先设置 DeepSeek API Key:")
        print("  方式 1: 设置环境变量 DEEPSEEK_API_KEY")
        print("  方式 2: 直接在本脚本中设置 api_key 变量")
        print("\n获取 API Key: https://platform.deepseek.com/api_keys")
        return

    # 测试连接
    if not test_llm_connection(api_key):
        print("\n✗ API 连接失败，请检查 API Key 是否正确")
        return

    # 3. 处理样本
    print("\n步骤 3: 处理样本数据")
    print("-"*80)

    sample_size = 10  # 先用小样本测试
    print(f"样本大小: {sample_size}\n")

    results_df = process_sample_batch(
        base_df=df_base,
        parts_df=df_parts,
        time_df=df_time,
        api_key=api_key,
        sample_size=sample_size
    )

    # 4. 显示结果
    print("\n步骤 4: 显示结果")
    print("-"*80)
    print(results_df.to_string())

    print("\n" + "="*80)
    print("✓ 测试完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
