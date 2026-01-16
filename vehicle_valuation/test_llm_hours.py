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
from utils.llm_structuring import process_sample_batch_concurrent
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

    # 3. 处理全部数据（并发版本）
    total_records = len(df_base)
    print("\n" + "="*80)
    print(f"步骤 2: 并发处理全部数据（共 {total_records:,} 条记录）")
    print("="*80 + "\n")

    # 并发配置（已优化：降低并发数，增加重试机制）
    max_workers = 5  # 并发线程数，降低以避免 API 速率限制
    print(f"⚠️  并发配置: {max_workers} 个线程（已优化以避免 API 限制）")
    print(f"预计速度: 约 {max_workers * 0.5} 条/秒（带重试机制）")
    print(f"预计耗时: 约 {total_records / (max_workers * 0.5) / 60:.1f} 分钟\n")

    # 询问确认
    confirm = input("是否继续？(输入 'yes' 确认): ").strip()

    if confirm.lower() != 'yes':
        print("\n处理已取消")
        return

    results_df = process_sample_batch_concurrent(
        base_df=df_base,
        parts_df=df_parts,
        time_df=df_time,
        api_key=api_key,
        sample_size=total_records,  # 处理全部数据
        max_workers=max_workers  # 并发线程数
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
