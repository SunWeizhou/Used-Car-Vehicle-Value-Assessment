#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保守重试脚本 - 单线程处理剩余失败记录

使用更保守的策略应对 API 限制：
- 单线程处理避免速率限制
- 更长的重试间隔（5秒、10秒、20秒）
- 每次请求后暂停1秒
"""

import sys
from pathlib import Path
import pandas as pd
import time

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.preprocessing import load_and_clean_data
from utils.llm_structuring import process_sample_batch_concurrent
import os


def main():
    """保守重试失败的记录"""
    print("\n" + "="*80)
    print("保守重试 - 单线程处理剩余失败记录")
    print("="*80 + "\n")

    # 1. 加载数据
    print("步骤 1: 加载数据\n")
    data_dir = Path(__file__).parent / "data"
    df_base, df_parts, df_time = load_and_clean_data(str(data_dir))

    # 2. 加载已有结果
    results_path = data_dir / "llm_parsed_results.csv"
    if not results_path.exists():
        print("❌ 未找到已有结果文件")
        return

    existing_results = pd.read_csv(results_path)
    print(f"✓ 已有结果: {len(existing_results)} 条记录\n")

    # 3. 筛选失败的记录
    failed_records = existing_results[existing_results['Severity'] == 'ERROR']
    print(f"步骤 2: 发现失败记录 {len(failed_records)} 条\n")

    if len(failed_records) == 0:
        print("✓ 没有需要重试的记录")
        return

    # 4. 获取 API Key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
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
        return

    # 5. 重试配置（保守策略）
    print("="*80)
    print(f"步骤 3: 保守重试失败记录（共 {len(failed_records)} 条）")
    print("="*80 + "\n")

    max_workers = 1  # 单线程
    max_retries = 5   # 更多重试次数
    print(f"⚠️  保守配置:")
    print(f"  - 单线程处理（避免 API 速率限制）")
    print(f"  - 重试次数: {max_retries} 次（指数退避）")
    print(f"  - 请求间隔: 1 秒（避免触发限制）")
    print(f"预计速度: 约 0.3-0.5 条/秒")
    print(f"预计耗时: 约 {len(failed_records) / 0.4 / 60:.1f} 分钟\n")

    confirm = input("是否继续？(输入 'yes' 确认): ").strip()
    if confirm.lower() != 'yes':
        print("\n处理已取消")
        return

    # 6. 获取需要重试的 ID
    retry_ids = failed_records['ID'].tolist()

    # 7. 只处理这些 ID
    retry_df = df_base[df_base['ID'].isin(retry_ids)].copy()

    print(f"\n开始保守重试 {len(retry_df)} 条失败记录...\n")

    # 8. 调用并发处理（单线程）
    results_df = process_sample_batch_concurrent(
        base_df=retry_df,
        parts_df=df_parts,
        time_df=df_time,
        api_key=api_key,
        sample_size=len(retry_df),
        max_workers=max_workers,  # 单线程
        max_retries=max_retries   # 更多重试
    )

    # 9. 合并结果
    print("\n" + "="*80)
    print("步骤 4: 合并结果")
    print("="*80 + "\n")

    # 删除旧的失败记录
    success_records = existing_results[existing_results['Severity'] != 'ERROR'].copy()

    # 合并成功记录和新结果
    all_results = pd.concat([success_records, results_df], ignore_index=True)

    # 按 ID 去重
    all_results = all_results.drop_duplicates(subset=['ID'], keep='last')

    # 保存
    all_results.to_csv(results_path, index=False, encoding='utf-8-sig')

    print(f"\n✓ 合并完成")
    print(f"  - 原成功记录: {len(success_records)} 条")
    print(f"  - 新处理记录: {len(results_df)} 条")
    print(f"  - 合并后总计: {len(all_results)} 条")
    print(f"  - 结果已保存至: {results_path}\n")

    # 统计
    print("【最终统计】")
    print(f"\n严重程度分布:")
    print(all_results['Severity'].value_counts())
    print(f"\n事件类型分布:")
    print(all_results['Event_Type'].value_counts())

    errors = all_results[all_results['Severity'] == 'ERROR']
    if len(errors) > 0:
        print(f"\n仍有失败记录: {len(errors)} 条")
        print("\n建议:")
        print("  1. 如果失败率仍高，可能需要检查 DeepSeek API 配额")
        print("  2. 可以稍后重试（避开 API 高峰期）")
        print("  3. 或者跳过这些记录，继续处理剩余数据")
    else:
        print("\n✓ 所有记录处理成功！")


if __name__ == "__main__":
    main()
