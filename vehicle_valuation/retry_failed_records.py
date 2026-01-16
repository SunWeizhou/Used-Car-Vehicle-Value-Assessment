#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重试失败的 LLM 处理记录

只重试之前处理失败的记录（ERROR标记）
"""

import sys
from pathlib import Path
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.preprocessing import load_and_clean_data
from utils.llm_structuring import process_sample_batch_concurrent
import os


def main():
    """重试失败的记录"""
    print("\n" + "="*80)
    print("重试失败的 LLM 处理记录")
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

    # 5. 重试配置
    print("="*80)
    print(f"步骤 3: 重试失败记录（共 {len(failed_records)} 条）")
    print("="*80 + "\n")

    max_workers = 5
    print(f"⚠️  并发配置: {max_workers} 个线程")
    print(f"预计速度: 约 {max_workers * 0.5} 条/秒（带重试机制）")
    print(f"预计耗时: 约 {len(failed_records) / (max_workers * 0.5) / 60:.1f} 分钟\n")

    confirm = input("是否继续？(输入 'yes' 确认): ").strip()
    if confirm.lower() != 'yes':
        print("\n处理已取消")
        return

    # 6. 获取需要重试的 ID
    retry_ids = failed_records['ID'].tolist()

    # 7. 只处理这些 ID
    # 先从原数据中筛选这些记录
    retry_df = df_base[df_base['ID'].isin(retry_ids)].copy()

    print(f"\n开始重试 {len(retry_df)} 条失败记录...\n")

    # 8. 调用并发处理（只处理这些记录）
    results_df = process_sample_batch_concurrent(
        base_df=retry_df,  # 只传入需要重试的记录
        parts_df=df_parts,
        time_df=df_time,
        api_key=api_key,
        sample_size=len(retry_df),
        max_workers=max_workers
    )

    # 9. 合并结果（从原结果中删除失败的，添加新的）
    print("\n" + "="*80)
    print("步骤 4: 合并结果")
    print("="*80 + "\n")

    # 删除旧的失败记录
    success_records = existing_results[existing_results['Severity'] != 'ERROR'].copy()

    # 合并成功记录和新结果
    all_results = pd.concat([success_records, results_df], ignore_index=True)

    # 按 ID 去重（保留后面的，即新结果）
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
        print("错误原因统计:")
        print(errors['Reasoning'].value_counts().head(5))
    else:
        print("\n✓ 所有记录处理成功！")


if __name__ == "__main__":
    main()
