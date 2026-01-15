#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 LLM 工时辅助判定功能

处理 10 条样本，验证 LLM 是否正确理解和使用工时数据
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.preprocessing import load_and_clean_data
from utils.llm_structuring import process_sample_batch
import os
from pathlib import Path


def main():
    """测试 LLM 工时辅助判定"""
    print("\n" + "="*80)
    print("LLM 工时辅助判定测试")
    print("="*80 + "\n")

    # 1. 加载数据
    print("步骤 1: 加载数据\n")
    data_dir = Path(__file__).parent / "data"
    df_base, df_parts, df_time = load_and_clean_data(str(data_dir))

    # 2. 获取 API Key
    api_key = os.environ.get("DEEPSEEK_API_KEY")

    # 尝试从 .env 文件加载
    if not api_key:
        env_file = project_root / ".env"
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('DEEPSEEK_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        os.environ['DEEPSEEK_API_KEY'] = api_key
                        break

    if not api_key:
        print("\n⚠️  未找到 DEEPSEEK_API_KEY")
        print("\n请输入您的 DeepSeek API Key (或按 Enter 跳过测试):")
        api_key = input("> ").strip()

        if not api_key:
            print("\n测试已取消")
            return

    # 3. 处理 10 条样本
    print("\n" + "="*80)
    print("步骤 2: 处理 10 条样本（带工时辅助判定）")
    print("="*80 + "\n")

    results_df = process_sample_batch(
        base_df=df_base,
        parts_df=df_parts,
        time_df=df_time,
        api_key=api_key,
        sample_size=10  # 只处理 10 条
    )

    # 4. 展示详细结果
    print("\n" + "="*80)
    print("步骤 3: 详细结果分析")
    print("="*80 + "\n")

    print("\n【推理过程展示】")
    print("所有记录的推理过程已在上方实时显示。\n")

    print("【汇总表】")
    print("\n" + "="*100)
    for idx, row in results_df.iterrows():
        print(f"\n记录 {row['ID']}:")
        print(f"  事件类型: {row['Event_Type']}")
        print(f"  系统: {row['System']}")
        print(f"  严重程度: {row['Severity']}")
        print(f"  推理: {row['Reasoning']}")
    print("="*100 + "\n")

    print("\n【严重程度分布】")
    print(results_df['Severity'].value_counts())

    print("\n【事件类型分布】")
    print(results_df['Event_Type'].value_counts())


if __name__ == "__main__":
    main()
